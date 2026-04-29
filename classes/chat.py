import streamlit as st
from openai import OpenAI
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import (
        GPT_BASE,
        GPT_KEY,
        GPT_CHAT_MODEL,
        GPT_SUPPORTS_REASONING,
        GPT_AVAILABLE_REASONING_EFFORTS,
        GPT_SUPPORTS_TEMPERATURE,
    )

from classes.description import (
    PlayerDescription,
    CountryDescription,
    PersonDescription,
    DefenderDescription,
    CBPairingDescription,
    CBReplacementDescription,
)
from classes.embeddings import PlayerEmbeddings, CountryEmbeddings, PersonEmbeddings

from classes.visual import Visual, DistributionPlot, DistributionPlotPersonality

import utils.sentences as sentences
from utils.gemini import convert_messages_format


class Chat:
    function_names = []

    def __init__(self, chat_state_hash, state="empty"):

        if (
            "chat_state_hash" not in st.session_state
            or chat_state_hash != st.session_state.chat_state_hash
        ):
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state
        if isinstance(self, PlayerChat):
            self.name = self.player.name
        elif isinstance(self, CBPairingChat):
            self.name = f"{self.player_a.name} & {self.player_b.name}"
        elif isinstance(self, PersonChat):
            self.name = self.person.name
        else:
            pass

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state

    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible=True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    # def get_input(self):
    #     """
    #     Get input from streamlit."""

    #     if x := st.chat_input(
    #         placeholder=f"What else would you like to know about {self.player.name}?"
    #     ):
    #         if len(x) > 500:
    #             st.error(
    #                 f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
    #             )

    #         self.handle_input(x)

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages.
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevant information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})

        messages.append(
            {
                "role": "user",
                "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```",
            }
        )

        # Remove all items in messages where content is not a string
        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        # Show the messages in an expander
        st.expander("Chat transcript", expanded=False).write(messages)

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_1.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
            if stream:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                        stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        stream=True,
                    )

                def streamed_chunks():
                    for event in response_stream:
                        if event.type == "response.output_text.delta":
                            yield event.delta

                answer = streamed_chunks()
            else:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                    )
                else:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                    )

                answer = response.output_text
        message = {"role": "assistant", "content": answer}

        # Add the returned value to the messages.
        self.messages_to_display.append(message)

    def display_content(self, content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try:
                content.show()
            except:
                try:
                    st.write(content.get_string())
                except:
                    raise ValueError(
                        f"Message content of type {type(content)} not supported."
                    )

    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns.
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        # st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message_block = st.chat_message(name=key, avatar=avatar)
            with message_block:
                for message in group:
                    content = message["content"]
                    if isinstance(content, GeneratorType):
                        final_text = st.write_stream(content)
                        message["content"] = final_text
                    else:
                        self.display_content(content)

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state


class SingleCBChat(Chat):
    def __init__(self, chat_state_hash, player, df, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.df = df
        self.name = player.name
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """Get input from streamlit."""
        if x := st.chat_input(
            placeholder=f"Ask about {self.player.name}'s defensive profile..."
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )
            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """Instruction for the agent."""
        return [
            {"role": "system", "content": "You are a defensive tactics analyst specializing in centre-back evaluation."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user analyzing a single centre-back. "
                    "You will receive relevant information about the player's ground duel quality metrics and then be asked to provide a response. "
                    "When responding to the user, speak directly to them and focus on this individual player's defensive profile. "
                    "You are a football analyst specialising in evaluating centre backs."
                    "You provide clear, concise, and data-driven descriptions of individual defenders based on their ground duel performance."
                    "You interpret statistical summaries to describe a player's defensive style, strengths, weaknesses, and overall effectiveness. "
                ),
            },
        ]

    def get_relevant_info(self, query):
        """Get relevant information about the player."""
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = f"Here is a ground duel quality analysis of {self.player.name}:\n\n"
        desc = DefenderDescription(self.player)
        ret_val += desc.synthesize_text() + "\n\n"

        results = self.embeddings.search(query, top_n=3)
        ret_val += "Here is relevant information for answering the question:\n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += (
            f"\n\nIf none of this information is relevant to the user's query, "
            f"remind them that this chat can answer questions about {self.player.name}'s "
            f"defensive quality, ground duel metrics, and how they compare to other centre-backs."
        )

        return ret_val


class CBPairingChat(Chat):
    def __init__(self, chat_state_hash, player_a, player_b, df, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player_a = player_a
        self.player_b = player_b
        self.df = df
        self.name = f"{player_a.name} & {player_b.name}"
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """Get input from streamlit."""
        if x := st.chat_input(
            placeholder=f"Ask about the {self.player_a.name} and {self.player_b.name} pairing..."
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )
            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """Instruction for the agent."""
        first_messages = [
            {"role": "system", "content": "You are a defensive tactics analyst specializing in center-back partnerships."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user analyzing a center-back pairing. "
                    f"The user has selected {self.player_a.name} and {self.player_b.name} as a defensive partnership. "
                    "You will receive relevant information about both players' ground duel quality metrics and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them and focus on how the two defenders complement or contrast with each other as a pairing. "
                    "Use the information provided before the query to provide 2-3 sentence answers about the partnership dynamics. "
                    "Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):
        """Get relevant information about both players in the pairing."""
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a comparison of the two center-backs in the pairing:\n\n"
        
        # Add description of both players
        ret_val += f"**{self.player_a.name}:**\n"
        desc_a = DefenderDescription(self.player_a)
        ret_val += desc_a.synthesize_text() + "\n\n"
        
        ret_val += f"**{self.player_b.name}:**\n"
        desc_b = DefenderDescription(self.player_b)
        ret_val += desc_b.synthesize_text() + "\n\n"

        # Add partnership context
        ret_val += f"**Partnership Context:**\n"
        ret_val += f"These two players can be analyzed as a defensive pairing in terms of complementary strengths and potential weaknesses. "
        ret_val += f"Consider how their duel quality metrics might work together in a partnership.\n\n"

        # Search for relevant information using embeddings
        results = self.embeddings.search(query, top_n=3)
        ret_val += "Here is relevant information for answering the question:\n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += (
            f"\n\nIf none of this information is relevant to the user's query, "
            f"remind them that this chat can answer questions about the defensive partnership between "
            f"{self.player_a.name} and {self.player_b.name}, including their complementary strengths, "
            f"how they compare to each other, and their combined defensive quality."
        )

        return ret_val


class CBAnalystChat(Chat):
    """
    Unified CB analyst chat. Uses OpenAI tool-calling so the LLM acts as a
    router, selecting one of three tools based on the user's question.

    All three tools are always available. The system prompt guides the LLM
    to use only pair/replacement tools when player_b is provided.
    """

    _TOOLS = [
        {
            "type": "function",
            "name": "get_player_summary",
            "description": (
                "Returns a full wordalised profile of Player A across ground duel, "
                "aerial duel, and ball-playing quality with sub-metric commentary. "
                "Use for questions about a single player's strengths, weaknesses, or overall rating."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "get_pair_evaluation",
            "description": (
                "Evaluates how Player A and Player B work together as a CB pairing. "
                "Covers complementarity, shared weaknesses, and combined profile. "
                "Use for questions about how two players work together or whether they are too similar."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "get_better_partner",
            "description": (
                "Suggests the best alternative CB partners to complement Player A. "
                "Identifies Player A's weakest quality dimension and ranks candidates "
                "strongest in that area. Use for questions about finding a better partner or replacement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quality_focus": {
                        "type": "string",
                        "enum": ["ground_duel", "aerial_duel", "ball_playing"],
                        "description": (
                            "Which quality dimension to prioritise. "
                            "Infer from the query if mentioned; otherwise omit to auto-detect."
                        ),
                    }
                },
                "required": [],
            },
        },
    ]

    _DESCRIBE_PATH    = "data/describe/Defender_Pair.xlsx"
    _GPT_EXAMPLE_PATH = "data/gpt_examples/Defenders_pair_example.xlsx"

    def __init__(self, chat_state_hash, player_a, player_b=None,
                 gd_df=None, ad_df=None, bp_df=None, state="empty"):
        self.player_a = player_a
        self.player_b = player_b

        # Pre-load Q&A domain knowledge (wordalisation step 2)
        try:
            df = pd.read_excel(self._DESCRIBE_PATH)
            self._domain_qa = [
                msg
                for _, row in df.iterrows()
                if pd.notna(row.get("user")) and pd.notna(row.get("assistant"))
                for msg in [
                    {"role": "user",      "content": str(row["user"])},
                    {"role": "assistant", "content": str(row["assistant"])},
                ]
            ]
        except FileNotFoundError:
            self._domain_qa = []

        # Build merged all-player DataFrame with all z-score columns
        self.all_players_df = (
            gd_df[["player.id", "player.name", "duel_quality",
                   "z_duel_success_rate", "z_possession_win_rate",
                   "z_discipline", "z_card_discipline",
                   "z_duels_per90", "z_interceptions_per90"]]
            .merge(
                ad_df[["player.id", "aerial_duel_quality",
                       "z_aerial_duel_success_rate", "z_aerial_duels_per90",
                       "z_aerial_won_duel_per90"]],
                on="player.id", how="inner",
            )
            .merge(
                bp_df[["player.id", "ball_playing_quality",
                       "z_Accuracy_Adjusted_Risk_per_Pass", "z_xT_per_Pass",
                       "z_FT_Entry_Passes_per_90", "z_xT_via_Carries_per_90"]],
                on="player.id", how="inner",
            )
        )

        self.name = (
            f"{player_a.name} & {player_b.name}"
            if player_b is not None
            else player_a.name
        )
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        placeholder = (
            f"Ask about {self.player_a.name} & {self.player_b.name}..."
            if self.player_b is not None
            else f"Ask about {self.player_a.name}..."
        )
        if x := st.chat_input(placeholder=placeholder):
            if len(x) > 500:
                st.error("Message too long — keep it under 500 characters.")
            else:
                self.handle_input(x, stream=True)

    def instruction_messages(self):
        if self.player_b is None:
            context = (
                f"The user is analysing {self.player_a.name} as an individual centre-back. "
                "Only get_player_summary is relevant — always use it to answer questions about this player. "
                "Do not call get_pair_evaluation or get_better_partner as no second player is selected."
            )
        else:
            context = (
                f"Player A is {self.player_a.name}. Player B is {self.player_b.name}. "
                "Use get_player_summary for questions about an individual player's profile. "
                "Use get_pair_evaluation for questions about how they work together. "
                "Use get_better_partner for questions about finding a better partner or replacement."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Centre-Back Pairing Analyst. "
                    "You answer ONLY questions about these metrics: "
                    "ground duel quality, aerial duel quality, and ball-playing quality, "
                    "including their sub-metrics (duel success rate, possession win rate, "
                    "discipline, aerial success rate, pass safety, xT per pass, etc.). "
                    "\n\n**When to use tools:**\n"
                    "- Call a tool when asked about a SPECIFIC PLAYER's statistics, profile, strengths, weaknesses, or ratings.\n"
                    "- Answer directly WITHOUT tools for GENERAL questions about metric definitions, concepts, or how analysis works (e.g., 'what is a ground duel?', 'how do you measure aerial quality?').\n"
                    "\n\n"
                    "**When answering:**\n"
                    "- If a 'Pre-written answer from wordalisation' is provided in the context, use similar phrasing.\n"
                    "- If a tool was called, combine the statistical data with any 'Relevant wordalisation context' to explain WHAT the metrics mean.\n"
                    "- Always use the wordalisation definitions when describing qualities and sub-metrics.\n"
                    "- Answer in 3-4 concise sentences that blend data with conceptual explanations.\n"
                    "- If the user asks anything outside these metrics, respond with: "
                    "'I can only answer questions about on-pitch metrics for Defenders: ground duels, aerial duels, "
                    "and ball playing. Please ask about one of those.'\n\n"
                    + context
                ),
            }
        ]
        if self._domain_qa:
            messages += [
                {"role": "user",      "content": "First, could you answer some questions about CB analysis for me?"},
                {"role": "assistant", "content": "Sure!"},
            ] + self._domain_qa
        return messages

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        """Two-pass tool-use loop: LLM selects tool → execute → LLM generates answer."""
        messages = self.instruction_messages() + self.messages_to_display.copy()
        self.messages_to_display.append({"role": "user", "content": input})
        
        # Get relevant wordalisation info for generic questions
        relevant_info = self.get_relevant_info(input)
        
        if relevant_info:
            # Inject wordalisation answer as context
            messages.append({"role": "user", "content": f"{relevant_info}\n\nUser question: {input}"})
        else:
            messages.append({"role": "user", "content": input})

        messages = [m for m in messages if isinstance(m.get("content"), str)]

        if USE_GEMINI:
            # Gemini path: no tool-calling support — call the most relevant tool directly
            default_tool = (
                "get_player_summary" if self.player_b is None else "get_pair_evaluation"
            )
            tool_result = self._call_tool(default_tool, {})
            messages.append({"role": "user", "content": f"Statistical context: {tool_result}\n\nUser: {input}"})
            import google.generativeai as genai
            from utils.gemini import convert_messages_format
            genai.configure(api_key=GEMINI_API_KEY)
            converted = convert_messages_format(messages)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted["system_instruction"],
            )
            chat = model.start_chat(history=converted["history"])
            response = chat.send_message(content=converted["content"])
            answer = response.text
        else:
            client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

            # Pass 1 — LLM selects a tool
            resp1 = client.responses.create(
                model=GPT_CHAT_MODEL,
                input=messages,
                tools=self._TOOLS,
            )

            # Check if a tool was called
            tool_call = next(
                (item for item in resp1.output if item.type == "function_call"),
                None,
            )

            if tool_call:
                fn_args     = json.loads(tool_call.arguments) if tool_call.arguments else {}
                tool_result = self._call_tool(tool_call.name, fn_args)

                # Pass 2 — append ALL output items from resp1 (reasoning models
                # emit a 'reasoning' item before the 'function_call' item; the API
                # requires both to be present together when replaying the history)
                messages.extend(resp1.output)
                messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_result,
                })
                
                # Inject wordalisation context for describing the metrics returned by the tool
                keywords = self._get_keywords_for_tool(tool_call.name, input)
                wordalisation_context = self.get_wordalisation_context_for_metrics(keywords)
                if wordalisation_context:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"{wordalisation_context}\n\n"
                            "Use these definitions when describing the player's qualities. "
                            "Answer in 3-4 concise sentences combining the statistical data with these conceptual explanations."
                        ),
                    })
                
                st.expander(
                    f"Chat transcript (tool: {tool_call.name})", expanded=False
                ).write(messages)
            else:
                # No tool called — answer directly (shouldn't normally happen)
                tool_result = None
                st.expander("Chat transcript", expanded=False).write(messages)

            if stream:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL, input=messages,
                        reasoning={"effort": reasoning_effort}, stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL, input=messages,
                        temperature=temperature, stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL, input=messages, stream=True,
                    )

                def streamed_chunks():
                    for event in response_stream:
                        if event.type == "response.output_text.delta":
                            yield event.delta

                answer = streamed_chunks()
            else:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    resp2 = client.responses.create(
                        model=GPT_CHAT_MODEL, input=messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    resp2 = client.responses.create(
                        model=GPT_CHAT_MODEL, input=messages,
                        temperature=temperature,
                    )
                else:
                    resp2 = client.responses.create(
                        model=GPT_CHAT_MODEL, input=messages,
                    )
                answer = resp2.output_text

        self.messages_to_display.append({"role": "assistant", "content": answer})

    def get_relevant_info(self, query):
        """
        Retrieve relevant wordalisation context for generic questions.
        Returns exact pre-written answers from domain Q&A pairs when available.
        """
        if not query:
            return ""
        
        # Check if query matches a pre-loaded Q&A pair (case-insensitive)
        query_lower = query.lower().strip().rstrip('?').rstrip('.')
        
        for i in range(0, len(self._domain_qa), 2):
            if i + 1 < len(self._domain_qa):
                q = self._domain_qa[i].get("content", "").lower().strip().rstrip('?').rstrip('.')
                a = self._domain_qa[i + 1].get("content", "")
                
                # Exact or close match
                if query_lower == q or query_lower in q or q in query_lower:
                    return f"Pre-written answer from wordalisation:\n{a}"
        
        return ""

    def get_wordalisation_context_for_metrics(self, keywords):
        """
        Extract wordalisation Q&A pairs relevant to specific metrics/qualities.
        Used to provide consistent explanations when describing player stats.
        
        Args:
            keywords: list of terms to search for (e.g., ['ground duel', 'aerial', 'ball playing'])
        
        Returns:
            String containing relevant Q&A pairs for context
        """
        if not self._domain_qa or not keywords:
            return ""
        
        relevant_pairs = []
        keywords_lower = [k.lower() for k in keywords]
        
        for i in range(0, len(self._domain_qa), 2):
            if i + 1 < len(self._domain_qa):
                q = self._domain_qa[i].get("content", "")
                a = self._domain_qa[i + 1].get("content", "")
                q_lower = q.lower()
                
                # Check if any keyword appears in the question
                if any(kw in q_lower for kw in keywords_lower):
                    relevant_pairs.append(f"Q: {q}\nA: {a}")
        
        if relevant_pairs:
            return "Relevant wordalisation context for describing these metrics:\n\n" + "\n\n".join(relevant_pairs[:5])
        return ""

    def _get_keywords_for_tool(self, tool_name, user_query):
        """
        Determine which metric keywords are relevant for the given tool and query.
        Returns list of terms to search for in wordalisation Q&A pairs.
        """
        # Base keywords for each tool
        base_keywords = {
            "get_player_summary": [
                "ground duel quality", "aerial duel quality", "ball playing quality",
                "ground duel", "aerial duel", "ball playing",
                "duel success", "possession win", "discipline",
                "aerial success", "pass safety", "xT", "carries"
            ],
            "get_pair_evaluation": [
                "ground duel quality", "aerial duel quality", "ball playing quality",
                "complement", "pairing", "similar", "weakness"
            ],
            "get_better_partner": [
                "ground duel quality", "aerial duel quality", "ball playing quality",
                "partner", "replacement", "complement"
            ],
        }
        
        keywords = base_keywords.get(tool_name, [])
        
        # Add query-specific keywords (detect which quality dimension is being asked about)
        query_lower = user_query.lower()
        if any(term in query_lower for term in ["aerial", "in the air", "heading"]):
            keywords = ["aerial duel quality", "aerial duel", "aerial success"] + keywords
        elif any(term in query_lower for term in ["ground", "tackling", "defending"]):
            keywords = ["ground duel quality", "ground duel", "duel success"] + keywords
        elif any(term in query_lower for term in ["ball", "passing", "distribution", "playing"]):
            keywords = ["ball playing quality", "ball playing", "pass", "xT"] + keywords
        
        return keywords

    def _call_tool(self, name: str, args: dict) -> str:
        """Execute the named tool and return its synthesized player data."""
        if name == "get_player_summary":
            return DefenderDescription(self.player_a).synthesize_text()
        elif name == "get_pair_evaluation":
            if self.player_b is None:
                return "No second player selected. Please select Player B in the sidebar to evaluate a pairing."
            return CBPairingDescription(self.player_a, self.player_b).synthesize_text()
        elif name == "get_better_partner":
            return CBReplacementDescription(
                self.player_a,
                self.all_players_df,
                quality_focus=args.get("quality_focus"),
            ).synthesize_text()
        return "Tool not found."


class PlayerChat(Chat):
    def __init__(self, chat_state_hash, player, players, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.players = players
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.player.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are a UK-based football scout."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a football scouting platform. "
                    f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the player in terms of data: \n\n"
        description = PlayerDescription(self.player)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a player's statistics and what they mean for how they play football."
        ret_val += "The user can select the player they are interested in using the menu to the left."

        return ret_val


class WVSChat(Chat):
    def __init__(
        self,
        chat_state_hash,
        country,
        countries,
        description_dict,
        thresholds_dict,
        state="empty",
    ):
        # TODO:
        self.embeddings = CountryEmbeddings()
        self.country = country
        self.countries = countries
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.country.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        # TODO: Update first_messages
        first_messages = [
            {"role": "system", "content": "You are a researcher."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a data analysis platform. "
                    f"The user has selected the country {self.country.name}, and the conversation will be about different core value measured in the World Value Survey study. "
                    # "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the country in terms of data: \n\n"
        description = CountryDescription(
            self.country, self.description_dict, self.thresholds_dict
        )
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevant to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a country's core values."
        ret_val += "The user can select the country they are interested in using the menu to the left."

        return ret_val


class PersonChat(Chat):
    def __init__(self, chat_state_hash, person, persons, state="empty"):
        self.embeddings = PersonEmbeddings()
        self.person = person
        self.persons = persons
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are a recruiter."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of personality test platform. "
                    f"The user has selected the person {self.person.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the person in terms of data: \n\n"
        description = PersonDescription(self.person)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about person's statistics and what they mean about their personality."
        ret_val += "The user can select the persons they are interested in using the menu to the left."

        return ret_val

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.person.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)
