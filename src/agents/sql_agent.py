from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
import os
import time

_cached_toolkit = None
_cached_prompt = None
_cache_last_updated = None
CACHE_REFRESH_INTERVAL = 900
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_language: str
    input: str

class SQLAgent:
    def __init__(self, llm):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.graph = self._get_graph_executor()
    
    def _get_graph_executor(self):
        builder = StateGraph(State)
        db_agent = self._create_reasoning_node()
        builder.add_node("agent", db_agent)
        builder.add_node("notfound", self._not_found)

        def relevant_answer(state):
            if "no results found" in state["messages"][-1].content.lower():
                return "notfound"
            return END

        builder.add_conditional_edges(
            "agent",
            relevant_answer
        )
        builder.set_entry_point("agent")
        builder.add_edge("notfound", END)
        return builder.compile()
    
    def _not_found(self, state):
        translate_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        rephrase_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        translation_prompt = PromptTemplate.from_template(
            "Translate this to {language} language:\n\n"
            "I don't have information about {topic}, but I can assist you with questions about the Women's Football Eurocup 2025."
        )

        rephrase_prompt = PromptTemplate.from_template(
            "Rewrite the following question as a noun phrase that describes its topic, like a short description:\n"
            "Question: {question}\n"
            "Description:"
        )

        rephrased = rephrase_llm.invoke(rephrase_prompt.format(question=state["input"])).content.strip()
        full_prompt = translation_prompt.format(language=state["question_language"], topic=rephrased)
        translated = translate_llm.invoke(full_prompt)
        return {"messages": [translated]}

    def _create_reasoning_node(self):
        tools, prompt = self._setup_sql_toolkit()
        agent = create_openai_functions_agent(llm=self.llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=tools, max_iterations=15)

        def run_agent(state: State) -> dict:
            result = executor.invoke({"input": state["input"], "language": state["question_language"]})
            if "agent stopped due to iteration limit or time limit." in result["output"].lower():
                return {"messages": [AIMessage(content="Seems that I cannot answer that question, please try again with another question.")]}
            return {"messages": [AIMessage(content=result["output"])]}
        return run_agent
    
    def _setup_sql_toolkit(self):
        global _cached_toolkit, _cached_prompt, _cache_last_updated

        def initialize_cache():
            """Helper function to initialize the cache."""
            custom_table_info_dict = {
                    "players": """Table of players of the competition.
                        Columns:
                        - player_id (INT): The unique identifier for the player, primary key.
                        - player_name (TEXT or VARCHAR): The full name of the player.
                        - team_id (INT): The identifier for the team the player belongs to, references teams.team_id.
                        - age (INTEGER): The age of the player.
                        - player_position (TEXT or VARCHAR): The position of the player on the field.
                        - t_shirt_number (INT): The number in the team.
                        - club (TEXT or VARCHAR): The club where the player plays.
                        - is_captain (BOOLEAN): Whether the player is the captain of the team. Show this just if the player is the captain.
                        Example: To find a player named 'Aitana', you would query "SELECT * FROM players WHERE player_name ILIKE '%Aitana%'".
                        """,
                    "players_stats": """Table of stats of players.
                        Columns:
                        - id (INTEGER or BIGINT): The primary key for the stats entry.
                        - player_id (INT): The identifier of the player these stats belong to, references players.player_id.
                        - goals (INT): Number of goals scored by the player.
                        - assists (INT): Number of assists made by the player.
                        - penalties (INT): Number of penalties made by the player.
                        - matches_played (INT): Number of matches played.
                        - minutes_played (INT): Number of minutes played.
                        - yellow_cards (INT): Number of yellow cards received.
                        - red_cards (INT): Number of red cards received.
                        When asked about a player's performance, always join `players_stats` with `players` and return all available stats for the player.
                        Example: How is aitana performance? you would query select * from players_stats join players on players.player_id = players_stats.player_id WHERE players.player_name ilike '%Aitana%'".
                        Example: What can you say about Aitana?  you would query select * from players_stats join players on players.player_id = players_stats.player_id WHERE players.player_name ilike '%Aitana%'".
                        """,
                    "teams": """Table of teams of the competition.
                        Columns:
                        - team_id (INT): The unique identifier for the team, primary key.
                        - country (TEXT or VARCHAR): The name of the team. The full name of the team/country, preferred for display over team_id.
                        - coach (TEXT or VARCHAR): The name of the coach of the team.
                        Example: To find the coach of a team, you would query "SELECT coach FROM teams WHERE country = 'Spain'".
                        """,
                    "groups": """Table of groups of the competition.
                        Columns:
                        - group_id (INT): The unique identifier for the group, primary key.
                        - group_name (TEXT or VARCHAR): The name of the group.
                        Example: To find the group where Spain plays, you would query "SELECT groups.group_id, groups.group_name FROM group_standings JOIN teams ON teams.team_id = group_standings.team_id JOIN groups on groups.group_id = group_standings.group_id  WHERE teams.country = 'Spain'".
                        """,
                    "stadiums": """Table of stadiums of the competition.
                        Columns:
                        - stadium_id (INT): The unique identifier for the stadium, primary key.
                        - stadium_name (TEXT or VARCHAR): The name of the stadium.
                        - city (TEXT or VARCHAR): The city of the stadium.
                        """,
                    "competition_stages": """Table of competition stages.
                        Columns:
                        - stage_id (INT): The unique identifier for the stage, primary key.
                        - stage_name (TEXT or VARCHAR): The name of the stage. Example: "Group Stage", "Quarter Finals", "Semi Finals", "Final".
                        """,
                    "matches": """Table of matches of the competition.
                        **CRITICAL INSTRUCTION:** When querying matches, you **MUST ALWAYS** JOIN with the 'teams' table (on home_team_id and away_team_id to teams.team_id) to select the `teams.country` for display, and **MUST ALWAYS** JOIN with the 'stadiums' table (on stadium_id to stadiums.stadium_id) to select `stadiums.stadium_name`. Do not return team_ids or stadium_ids in the final SELECT if names are available.
                        Columns:
                        - match_id (INT): The unique identifier for the match, primary key.
                        - group_id (INT): The identifier for the group of the match, references groups.group_id.
                        - stage_id (INT): The identifier for the stage of the match, references competition_stages.stage_id.
                        - home_team_id (INT): The identifier for the first team, references teams.team_id.
                        - away_team_id (INT): The identifier for the second team, references teams.team_id.
                        - home_score (INT): The score of the first team.
                        - away_score (INT): The score of the second team.
                        - match_datetime (DATE or TIMESTAMP): The date and time of the match.
                        Example: To find all matches in a Semi-final stage, you would query "SELECT * FROM matches
                                            JOIN competition_stages on competition_stages.stage_id = matches.stage_id
                                            LEFT JOIN teams t1 on t1.team_id = matches.home_team_id
                                            LEFT JOIN teams t2 on t2.team_id = matches.away_team_id
                                            JOIN stadiums on stadiums.stadium_id = matches.stadium_id
                                            WHERE competition_stages.stage_name = 'Semi-final'".
                        Example: To find all matches of the Group C, you would query "SELECT * FROM matches
                                            JOIN groups on groups.group_id = matches.group_id
                                            JOIN teams t1 on t1.team_id = matches.home_team_id
                                            JOIN teams t2 on t2.team_id = matches.away_team_id
                                            JOIN stadiums on stadiums.stadium_id = matches.stadium_id
                                            WHERE groups.group_name = 'C'".
                        Example: To find all matches of the Spain, you would query "SELECT * FROM matches
                                        JOIN groups on groups.group_id = matches.group_id
                                        JOIN teams t1 on t1.team_id = matches.home_team_id
                                        JOIN teams t2 on t2.team_id = matches.away_team_id
                                        JOIN stadiums on stadiums.stadium_id = matches.stadium_id
                                        WHERE t1.country = 'Spain' or t2.country = 'Spain'".
                        Example: Find the matches of today, you would query  "SELECT * FROM matches
                                        JOIN groups on groups.group_id = matches.group_id
                                        JOIN teams t1 on t1.team_id = matches.home_team_id
                                        JOIN teams t2 on t2.team_id = matches.away_team_id
                                        JOIN stadiums on stadiums.stadium_id = matches.stadium_id
                                        WHERE matches.match_datetime >= CURRENT_DATE AND matches.match_datetime < CURRENT_DATE + INTERVAL '1 day'".
                        Example: List remaining matches for all teams in Group C, you would query  "SELECT matches.match_id, matches.match_datetime AS date, teams_a.country AS team_a, teams_b.country AS team_b                                
                                        FROM matches                                                                                                                                 
                                        JOIN teams t1 ON matches.home_team_id = t1.team_id                                                                              
                                        JOIN teams t2 ON matches.away_team_id = t2.team_id
                                        JOIN groups on groups.group_id = matches.group_id                                                                        
                                        WHERE groups.group_name = 'C' AND matches.match_datetime > CURRENT_DATE 
                                        ORDER BY matches.match_datetime ASC;".
                        """,
                        "group_standings": """Table of group standings of the competition.
                        Columns:
                        - position_id (INT): The unique identifier for the match, primary key.
                        - group_id (INT): The identifier for the group of the match, references groups.group_id.
                        - team_id (INT): The identifier for the team, references teams.team_id.
                        - points (INT): The points of the team.
                        - matches_played (INT): The number of matches played by the team.
                        - wins (INT): The number of wins of the team.
                        - draws (INT): The number of draws of the team.
                        - losses (INT): The number of losses of the team.
                        - goals_for (INT): The number of goals for the team.
                        - goals_against (INT): The number of goals against the team.
                        - goal_difference (INT): The goal difference of the team.
                        - group_position (INT): The position of the team in the group.
                        Example: Describe the group B, you would query "SELECT * FROM group_standings
                                            JOIN groups on groups.group_id = group_standings.group_id
                                            JOIN teams ON teams.team_id = group_standings.team_id
                                            WHERE groups.group_name = 'B'
                                            ORDER BY group_standings.group_position DESC".
                        Example: What is the position of Spain "SELECT group_position FROM group_standings
                                            JOIN groups on groups.group_id = group_standings.group_id
                                            JOIN teams ON teams.team_id = group_standings.team_id
                                            WHERE teams.country ilike '%spain%'".
                        Example: Retrieve the current standings for Group B, you would query "SELECT * FROM group_standings
                                            JOIN groups on groups.group_id = group_standings.group_id
                                            JOIN teams ON teams.team_id = group_standings.team_id
                                            WHERE groups.group_name = 'B'
                                            ORDER BY group_standings.group_position DESC".
                        """,
                        "team_match_stats": """Table of match stats of the competition
                        Columns:
                        - team_id (INT),
                        - possession_percent (INT): print it with a percentage sign, e.g. 55%.
                        - shots (INT),
                        - shots_on_target (INT),
                        - passes (INT),
                        - accurate_passes (INT): print it with a percentage sign, e.g. 55%.
                        - fouls (INT),
                        - corners (INT),
                        - offsides (INT),
                        Example: what are the stats of the match between Spain and Portugal, you would query "
                            SELECT 
                                m.match_id,
                                m.date,
                                m.venue,
                                t.country,
                                ts.possession_percent,
                                ts.shots,
                                ts.shots_on_target,
                                ts.passes,
                                ts.accurate_passes,
                                ts.fouls,
                                ts.corners,
                                ts.offsides,
                                ts.saves
                            FROM matches m
                            JOIN team_match_stats ts ON m.match_id = ts.match_id
                            JOIN teams t ON ts.team_id = t.team_id
                            WHERE 
                                (m.home_team_id = (SELECT team_id FROM teams WHERE country ilike '%spain%') AND
                                m.away_team_id = (SELECT team_id FROM teams WHERE country ilike '%portugal%'))
                            OR
                                (m.home_team_id = (SELECT team_id FROM teams WHERE country ilike '%portugal%') AND
                                m.away_team_id = (SELECT team_id FROM teams WHERE country ilike '%spain%'));
                            "
                        """,
                        "historical_matches": """Table of historical match scores
                        Columns:
                        - match_id (INT): The unique identifier for the match, primary key.
                        - home_team_id (INT): The identifier for the first team, references teams.team_id.
                        - away_team_id (INT): The identifier for the second team, references teams.team_id.
                        - home_score (INT): The score of the first team.
                        - away_score (INT): The score of the second team.
                        - match_datetime (DATE or TIMESTAMP): The date and time of the match.
                        Example: how were the previous matches between Spain and Portugal before of the euro or what are the previous matches between Spain and Portugal before of the euro, you would query "
                            SELECT 
                                hm.match_id,
                                th.country AS home_team,
                                hm.home_score,
                                ta.country AS away_team,
                                hm.away_score,
                                hm.match_datetime
                            FROM historical_matches hm
                            JOIN teams th ON hm.home_team_id = th.team_id
                            JOIN teams ta ON hm.away_team_id = ta.team_id
                            WHERE 
                                (th.country ilike '%spain%' AND ta.country ilike '%portugal%')
                                OR
                                (th.country ilike '%portugal%' AND ta.country ilike '%spain%');
                            " and you would do a summary with the total of victories, losses and draws for each team.
                        """,
                }

            system_message = """
                        You are an expert PostgreSQL assistant. Your primary goal is to generate a single, comprehensive, syntactically correct SQL query to answer the user's question, prioritizing human-readable information.
                        **IT IS CRUCIAL AND MANDATORY** to fetch human-readable names instead of IDs whenever names are available. For example, you **MUST ALWAYS** select team names from the 'teams' table (using `teams.country`) and stadium names from the 'stadiums' table (using `stadiums.stadium_name`). **DO NOT** return raw IDs like `home_team_id` or `stadium_id` in the final SELECT statement if the corresponding name can be joined and selected.
                        MANDATORY: For every SQL question, ALWAYS invoke `sql_db_list_tables` and `sql_db_schema` to retrieve the exact table structure. Never assume table schemas or column names.
                        When asked about matches, you **MUST** ensure your query left joins with the 'teams' table (aliasing as t1 for home_team and t2 for away_team if necessary) to get 'country' for both home and away teams, and also join with the 'stadiums' table to get 'stadium_name'. Use ILIKE for case-insensitive name searches and the value between %.
                        When asked about group standings, you **MUST** join with the 'teams' table to get the `teams.country` for display. Do not return `team_id` in the final SELECT if names are available.
                        When asked about a player's performance (for example, "How is Aitana performance?" or "what can you say about Aitana?"), you **MUST** join `players_stats` with `players` and return all available stats for the player (such as goals, assists, matches_played, yellow_cards, red_cards), using ILIKE for case-insensitive player name search and the value between %.
                        For example, if the user ask Retrieve the current standings for Group B:
                        SELECT * FROM group_standings
                        JOIN groups on groups.group_id = group_standings.group_id
                        JOIN teams ON teams.team_id = group_standings.team_id
                        WHERE groups.group_name = 'B'
                        ORDER BY group_standings.group_position DESC;

                        For example, if the user asks "What are the matches for Group B?", the query **MUST** be structured like this to include names:
                        SELECT m.match_datetime, ht.country AS home_team_name, at.country AS away_team_name, s.stadium_name
                        FROM matches m
                        JOIN groups g ON g.group_id = m.group_id
                        LEFT JOIN teams ht ON ht.team_id = m.home_team_id
                        LEFT JOIN teams at ON at.team_id = m.away_team_id
                        JOIN stadiums s ON s.stadium_id = m.stadium_id
                        WHERE g.group_name = 'B';

                        For example, if user asks "Who is in semi final?", you would query:
                        To find teams who will play in Semi-final stage, you would query "SELECT * FROM matches
                                            JOIN competition_stages on competition_stages.stage_id = matches.stage_id
                                            LEFT JOIN teams t1 on t1.team_id = matches.home_team_id
                                            LEFT JOIN teams t2 on t2.team_id = matches.away_team_id
                                            JOIN stadiums on stadiums.stadium_id = matches.stadium_id
                                            WHERE competition_stages.stage_name = 'Semi-final'".
                        For example, if the user asks "What are the stats of the match between Spain and Portugal?", you would query:
                                SELECT 
                                m.match_id,
                                m.date,
                                m.venue,
                                t.team_name,
                                ts.possession_percent,
                                ts.shots,
                                ts.shots_on_target,
                                ts.passes,
                                ts.accurate_passes,
                                ts.fouls,
                                ts.corners,
                                ts.offsides,
                                ts.saves
                            FROM matches m
                            JOIN team_match_stats ts ON m.match_id = ts.match_id
                            JOIN teams t ON ts.team_id = t.team_id
                            WHERE 
                                (m.home_team_id = (SELECT team_id FROM teams WHERE team_name = 'Spain') AND
                                m.away_team_id = (SELECT team_id FROM teams WHERE team_name = 'Portugal'))
                            OR
                                (m.home_team_id = (SELECT team_id FROM teams WHERE team_name = 'Portugal') AND
                                m.away_team_id = (SELECT team_id FROM teams WHERE team_name = 'Spain'));

                        For example, if the user asks "how were the previous matches between Spain and Portugal before of the euro?" or "what are the previous matches between Spain and Portugal before of the euro?, you would query:
                                SELECT 
                                hm.match_id,
                                th.team_name AS home_team,
                                hm.home_score,
                                ta.team_name AS away_team,
                                hm.away_score,
                                hm.match_datetime
                            FROM historical_matches hm
                            JOIN teams th ON hm.home_team_id = th.team_id
                            JOIN teams ta ON hm.away_team_id = ta.team_id
                            WHERE 
                                (th.team_name = 'Spain' AND ta.team_name = 'Portugal')
                                OR
                                (th.team_name = 'Portugal' AND ta.team_name = 'Spain');
                        For queries like "previous matches between [Team A] and [Team B]" or "matches before the Euro," always use the `historical_matches` table joined with `teams` for team names and do a summary with the total of victories, losses, and draws for each team.
                        A query that only returns IDs like `SELECT m.match_id, m.home_team_id, m.away_team_id FROM matches m ... WHERE g.group_name = 'B'` is **INCORRECT AND STRICTLY FORBIDDEN** if names can be retrieved.
                        **Never include IDs in the response.
                        **If a query returns no results, try rewriting the query using LEFT JOINs instead of regular JOINs (especially for teams and stadiums tables) to include matches even if some related data is missing.**
                        If the queries does not return any results, return "No results found", do not hallucinate.
                        IMPORTANT: Do not just describe the SQL query. Always call the appropriate tool to execute the SQL and return the results unless explicitly told to explain the query only.
                        Answer in {language} language.
                """
            db = SQLDatabase.from_uri(
                    os.getenv("POSTGRES_HOST"),
                    include_tables=["teams", "players", "players_stats", "groups", "stadiums", "competition_stages", "matches", "group_standings", "team_match_stats", "historical_matches"],
                    sample_rows_in_table_info=2,
                    custom_table_info=custom_table_info_dict
            )
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_message),
                    HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
            ])

            return toolkit.get_tools(), prompt
         
        try:
            # Check if cache needs to be refreshed
            current_time = time.time()
            if (_cached_toolkit and _cached_prompt and 
                _cache_last_updated and 
                current_time - _cache_last_updated < CACHE_REFRESH_INTERVAL):
                return _cached_toolkit, _cached_prompt
            
            # Update cache
            _cached_toolkit, _cached_prompt = initialize_cache()
            _cache_last_updated = current_time
            return _cached_toolkit, _cached_prompt
        except Exception as e:
            print(f"Error initializing SQL toolkit or prompt: {e}")
            _cached_toolkit, _cached_prompt = initialize_cache()
            _cache_last_updated = current_time
            return _cached_toolkit, _cached_prompt