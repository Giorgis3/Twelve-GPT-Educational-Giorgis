from abc import ABC, abstractmethod
from typing import List, Union, Dict

import pandas as pd

from openai import OpenAI


import utils.sentences as sentences
from utils.gemini import convert_messages_format

from classes.data_point import Player, Country, Person
from classes.data_source import PersonStat

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

import streamlit as st



class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis bot. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def get_messages_from_excel(
        self,
        paths: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        try:
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  
            print(e)
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
            )
        except (
            FileNotFoundError
        ) as e:  
            print(e)

        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following: ```{self.synthesized_text}```",
            }
        ]
        return messages

    def stream_gpt(self, temperature=1, reasoning_effort=None, stream=False):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.

        Yields:
            str
        """

        st.expander("Chat transcript", expanded=False).write(self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_0.json", "w") as f:
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
                        input=self.messages,
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        temperature=temperature,
                        stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
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
                        input=self.messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                        temperature=temperature,
                    )
                else:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=self.messages,
                    )

                answer = response.output_text

        return answer


class PlayerDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, player: Player):
        self.player = player
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football scout. "
                    "You provide succinct and to the point explanations about football players using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player = self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)

        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric + "_Z"])
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's playing style, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the player. "
            "The second sentence should describe the player's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]


class DefenderDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Defenders_example.xlsx"  # Can reuse or create Defender_examples.xlsx

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Defender_Pair.xlsx"]

    def __init__(self, player: Player):
        self.player = player
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a defensive tactics analyst. "
                    "You are a football analyst specialising in evaluating centre backs. "
                    "You provide concise, data-driven descriptions of defenders based on ground duel quality performance."
                    "You use the information given to you from the data and answers "
                    "When relevant, you also analyse how two centre backs complement each other as a pair, focusing on balance, roles, and defensive interaction."
                    "You base your analysis only on the information provided and do not introduce external knowledge."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about ground duel quality for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    # ── Sub-metric definitions shared across all CB description classes ─────────
    _GD_METRICS = {
        "z_possession_win_rate":  "possession win rate",
        "z_duel_success_rate":    "duel success rate",
        "z_interceptions_per90":  "interceptions per 90",
        "z_duels_per90":          "duels per 90",
        "z_discipline":           "discipline",
        "z_card_discipline":      "card discipline",
    }
    _AD_METRICS = {
        "z_aerial_duel_success_rate": "aerial success rate",
        "z_aerial_duels_per90":       "aerial duels per 90",
        "z_aerial_won_duel_per90":    "aerial duels won per 90",
    }
    _BP_METRICS = {
        "z_Accuracy_Adjusted_Risk_per_Pass": "pass safety",
        "z_xT_per_Pass":                     "xT per pass",
        "z_FT_Entry_Passes_per_90":          "final-third entry passes per 90",
        "z_xT_via_Carries_per_90":           "xT via carries per 90",
    }

    def synthesize_text(self):
        player = self.player
        m = player.ser_metrics
        name = player.name

        description = f"Here is a full quality profile of {name}.\n\n"

        # ── Ground Duel ──────────────────────────────────────────────────────
        description += "GROUND DUEL QUALITY:\n"
        for col, label in self._GD_METRICS.items():
            if col in m:
                description += f"{name} is {sentences.describe_level(m[col])} in {label} compared to other centre-backs. "
        description += "\n\n"

        # ── Aerial Duel ──────────────────────────────────────────────────────
        aerial_cols_present = [c for c in self._AD_METRICS if c in m]
        if aerial_cols_present:
            description += "AERIAL DUEL QUALITY:\n"
            for col in aerial_cols_present:
                description += f"{name} is {sentences.describe_level(m[col])} in {self._AD_METRICS[col]} compared to other centre-backs. "
            description += "\n\n"

        # ── Ball Playing ─────────────────────────────────────────────────────
        bp_cols_present = [c for c in self._BP_METRICS if c in m]
        if bp_cols_present:
            description += "BALL PLAYING QUALITY:\n"
            for col in bp_cols_present:
                description += f"{name} is {sentences.describe_level(m[col])} in {self._BP_METRICS[col]} compared to other centre-backs. "
            description += "\n\n"

        # ── Composite quality levels ─────────────────────────────────────────
        composite = []
        if "duel_quality" in m:
            composite.append(f"ground duel quality: {sentences.describe_level(m['duel_quality'])}")
        if "aerial_duel_quality" in m:
            composite.append(f"aerial duel quality: {sentences.describe_level(m['aerial_duel_quality'])}")
        if "ball_playing_quality" in m:
            composite.append(f"ball playing quality: {sentences.describe_level(m['ball_playing_quality'])}")
        if composite:
            description += "COMPOSITE QUALITY SCORES:\n"
            description += f"{name}: " + ", ".join(composite) + ".\n\n"

        return description

    def get_prompt_messages(self):
        name = self.player.name
        prompt = (
            "Please use the statistical description enclosed with ``` to do the following:\n\n"
            f"First, write a 3-sentence summary of {name}'s overall defensive profile covering all three quality dimensions "
            "(ground duels, aerial duels, and ball playing). "
            "Then answer each of the following questions. Use only the data provided.\n\n"
            f"Q1: What is {name}'s overall profile across all three quality dimensions? (1-2 sentences)\n\n"
            f"Q2: What are {name}'s main strengths? Call out specific sub-metrics by name. (1-2 sentences)\n\n"
            f"Q3: What are {name}'s weaknesses? Call out specific sub-metrics by name. (1-2 sentences)\n\n"
            f"Q4: How does {name} contribute in ground duels — activity and success rate? (1-2 sentences)\n\n"
            f"Q5: How does {name} perform aerially? (1-2 sentences)\n\n"
            f"Q6: How does {name} contribute in ball playing and build-up? (1-2 sentences)\n\n"
            f"Q7: How does {name} compare to other centre-backs in the league overall? (1-2 sentences)\n\n"
            "Format your response as:\n"
            "Summary: [3-sentence summary]\n\n"
            "Q: [question]\nA: [answer]\n"
        )
        return [{"role": "user", "content": prompt}]


class CBPairingDescription(Description):
    output_token_limit = 300

    @property
    def gpt_examples_path(self):
        return [] if not self.use_wordalisation else f"{self.gpt_examples_base}/Defenders_pair_example.xlsx"

    @property
    def describe_paths(self):
        return [] if not self.use_wordalisation else [f"{self.describe_base}/Defender_Pair.xlsx"]

    def __init__(self, player_a: Player, player_b: Player, df=None, use_wordalisation=True):
        self.player_a = player_a
        self.player_b = player_b
        self.df = df
        self.use_wordalisation = use_wordalisation
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a football analyst specialising in centre-back pair evaluation on ground duels and make replacements when asked. "
                    "Use only the statistical descriptions provided. "
                    "Base every conclusion directly on the listed metrics: possession win rate, duel success rate, "
                    "interceptions per 90, duels per 90, discipline, and card discipline. "
                    "Do not introduce external knowledge or infer qualities not clearly supported by the data. "
                    "If both players are weak, passive, or risky in the same area, state that clearly. "
                    "Write concise and grounded football interpretations."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about ground duel quality for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]
        return intro

    # ── Shared sub-metric definitions (same as DefenderDescription) ─────────
    _GD_METRICS = {
        "z_possession_win_rate":  "possession win rate",
        "z_duel_success_rate":    "duel success rate",
        "z_interceptions_per90":  "interceptions per 90",
        "z_duels_per90":          "duels per 90",
        "z_discipline":           "discipline",
        "z_card_discipline":      "card discipline",
    }
    _AD_METRICS = {
        "z_aerial_duel_success_rate": "aerial success rate",
        "z_aerial_duels_per90":       "aerial duels per 90",
        "z_aerial_won_duel_per90":    "aerial duels won per 90",
    }
    _BP_METRICS = {
        "z_Accuracy_Adjusted_Risk_per_Pass": "pass safety",
        "z_xT_per_Pass":                     "xT per pass",
        "z_FT_Entry_Passes_per_90":          "final-third entry passes per 90",
        "z_xT_via_Carries_per_90":           "xT via carries per 90",
    }

    def _describe_player(self, player) -> str:
        """One player's metrics across all three quality dimensions."""
        m = player.ser_metrics
        name = player.name
        out = f"{name}:\n"

        for col, label in self._GD_METRICS.items():
            if col in m:
                out += f"  {label}: {sentences.describe_level(m[col])}\n"
        for col, label in self._AD_METRICS.items():
            if col in m:
                out += f"  {label}: {sentences.describe_level(m[col])}\n"
        for col, label in self._BP_METRICS.items():
            if col in m:
                out += f"  {label}: {sentences.describe_level(m[col])}\n"

        # Composite scores
        for composite_col, composite_label in [
            ("duel_quality",        "ground duel quality"),
            ("aerial_duel_quality", "aerial duel quality"),
            ("ball_playing_quality","ball playing quality"),
        ]:
            if composite_col in m:
                out += f"  {composite_label}: {sentences.describe_level(m[composite_col])}\n"
        return out + "\n"

    def synthesize_text(self) -> str:
        a_name = self.player_a.name
        b_name = self.player_b.name

        description = (
            f"Here is a full quality profile of centre backs {a_name} and "
            f"{b_name} as a defensive pairing.\n\n"
        )
        description += self._describe_player(self.player_a)
        description += self._describe_player(self.player_b)

        return description

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        a, b = self.player_a.name, self.player_b.name
        prompt = (
            "Please use only the statistical description enclosed with ``` to analyse this centre-back pairing.\n\n"
            f"First, write a 3-sentence summary of the partnership between {a} and {b}.\n\n"
            "Sentence 1: overall defensive profile of the pair as a unit across all three quality dimensions.\n"
            "Sentence 2: how their qualities complement or overlap — name specific sub-metrics.\n"
            "Sentence 3: the main shared weakness or risk of the pairing — name specific sub-metrics.\n\n"
            "Then answer each of the following questions. Use only the data provided.\n\n"
            f"Q1: How do {a} and {b} compare in ground duel quality? Which sub-metrics differ most? (2-3 sentences)\n\n"
            f"Q2: How do {a} and {b} compare aerially? Are they complementary or similar? (1-2 sentences)\n\n"
            f"Q3: How do {a} and {b} compare in ball playing? What is the combined risk? (1-2 sentences)\n\n"
            f"Q4: What is the combined strength of this pairing? (1-2 sentences)\n\n"
            f"Q5: What is the main weakness of this pairing? Name specific sub-metrics. (1-2 sentences)\n\n"
            f"Q6: Are these two players too similar, or do they complement each other well? (1-2 sentences)\n\n"
            "Format your response as:\n"
            "Summary: [3-sentence summary]\n\nQ: [question]\nA: [answer]\n"
        )
        return [{"role": "user", "content": prompt}]


class CBReplacementDescription(Description):
    """
    Identify the weakest quality dimension for player_a and rank the best
    complementary partners from the full CB pool.
    """
    output_token_limit = 300

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Defenders_pair_example.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Defender_Pair.xlsx"]

    # ── Sub-metric definitions ───────────────────────────────────────────────
    _QUALITY_COLS = {
        "ground_duel":  ("duel_quality",         "ground duel quality"),
        "aerial_duel":  ("aerial_duel_quality",  "aerial duel quality"),
        "ball_playing": ("ball_playing_quality", "ball playing quality"),
    }
    _SUBMETRICS = {
        "ground_duel": {
            "z_possession_win_rate":  "possession win rate",
            "z_duel_success_rate":    "duel success rate",
            "z_interceptions_per90":  "interceptions per 90",
            "z_duels_per90":          "duels per 90",
            "z_discipline":           "discipline",
        },
        "aerial_duel": {
            "z_aerial_duel_success_rate": "aerial success rate",
            "z_aerial_duels_per90":       "aerial duels per 90",
            "z_aerial_won_duel_per90":    "aerial duels won per 90",
        },
        "ball_playing": {
            "z_Accuracy_Adjusted_Risk_per_Pass": "pass safety",
            "z_xT_per_Pass":                     "xT per pass",
            "z_FT_Entry_Passes_per_90":          "final-third entry passes per 90",
            "z_xT_via_Carries_per_90":           "xT via carries per 90",
        },
    }

    def __init__(self, player_a, all_players_df, quality_focus=None):
        """
        Parameters
        ----------
        player_a        : Player  — the CB being evaluated
        all_players_df  : pd.DataFrame — merged DataFrame with all quality +
                          z-score columns; must include player.id and player.name
        quality_focus   : str | None — "ground_duel" | "aerial_duel" |
                          "ball_playing"; auto-detected if None
        """
        self.player_a       = player_a
        self.all_players_df = all_players_df
        self.quality_focus  = quality_focus
        super().__init__()

    def _detect_weakest_quality(self):
        """Return the quality key whose composite z-score is lowest for player_a."""
        m = self.player_a.ser_metrics
        scores = {
            key: m[col]
            for key, (col, _) in self._QUALITY_COLS.items()
            if col in m
        }
        if not scores:
            return "ground_duel"
        return min(scores, key=scores.get)

    def synthesize_text(self) -> str:
        focus     = self.quality_focus or self._detect_weakest_quality()
        col, label = self._QUALITY_COLS.get(focus, ("duel_quality", "ground duel quality"))
        sub_map   = self._SUBMETRICS.get(focus, {})
        name_a    = self.player_a.name
        m_a       = self.player_a.ser_metrics

        # ── Describe player_a's weakness ────────────────────────────────────
        description = f"REPLACEMENT ANALYSIS for {name_a}.\n\n"

        if col in m_a:
            description += (
                f"{name_a}'s weakest quality dimension is {label} "
                f"({sentences.describe_level(m_a[col])}).\n"
            )
        description += f"Weakest sub-metrics in {label}:\n"
        for z_col, sub_label in sub_map.items():
            if z_col in m_a:
                description += f"  {sub_label}: {sentences.describe_level(m_a[z_col])}\n"
        description += "\n"

        # ── Rank top-5 complementary candidates ─────────────────────────────
        df = self.all_players_df
        if col not in df.columns or "player.name" not in df.columns:
            description += "No candidate data available.\n"
            return description

        others = df[df["player.name"] != name_a].copy()
        if col not in others.columns:
            return description

        top = others.nlargest(5, col)

        description += f"Top complementary partners ranked by {label}:\n"
        for _, row in top.iterrows():
            cand_name = row["player.name"]
            cand_composite = sentences.describe_level(row[col]) if pd.notna(row.get(col)) else "n/a"
            sub_parts = []
            for z_col, sub_label in sub_map.items():
                if z_col in row.index and pd.notna(row[z_col]):
                    sub_parts.append(f"{sub_label}: {sentences.describe_level(row[z_col])}")
            subs = ", ".join(sub_parts) if sub_parts else "no sub-metric data"
            description += f"  - {cand_name}: {label} overall {cand_composite}. {subs}\n"

        return description

    def get_intro_messages(self) -> List[Dict[str, str]]:
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a football recruitment analyst specialising in centre-back pairings. "
                    "Given a CB's weakest quality dimension and a ranked list of complementary "
                    "candidates, you recommend the best replacement partner and explain why they "
                    "address the gap. Base every conclusion on the data provided."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {"role": "user", "content": "First, could you answer some questions about CB pairings?"},
                {"role": "assistant", "content": "Sure!"},
            ]
        return intro

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        focus = self.quality_focus or self._detect_weakest_quality()
        _, label = self._QUALITY_COLS.get(focus, ("duel_quality", "ground duel quality"))
        name = self.player_a.name
        prompt = (
            f"Please use the statistical description enclosed with ``` to recommend the best "
            f"CB partner for {name}.\n\n"
            f"Q1: What is {name}'s weakest quality dimension, and which specific sub-metrics are lowest? "
            f"(2-3 sentences)\n\n"
            f"Q2: Which player from the candidate list would best complement {name} in {label}, "
            f"and why? Name the specific sub-metrics that make them a good fit. (2-3 sentences)\n\n"
            f"Q3: Which is the second-best option, and how does it compare to the top pick? (1-2 sentences)\n\n"
            f"Q4: What kind of pairing would {name} + the recommended partner form overall? "
            f"Any remaining shared weaknesses to be aware of? (1-2 sentences)\n\n"
            "Format your response as:\n"
            "Recommendation: [top pick + 1-2 sentence justification]\n\n"
            "Q: [question]\nA: [answer]\n"
        )
        return [{"role": "user", "content": prompt}]


class CountryDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/WVS_examples.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/WVS_qualities.xlsx"]

    def __init__(self, country: Country, description_dict, thresholds_dict):
        self.country = country
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict

        # read data/wvs/intermediate_data/relevant_questions.json
        with open("data/wvs/intermediate_data/relevant_questions.json", "r") as f:
            self.relevant_questions = json.load(f)

        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst and a social scientist. "
                    "You provide succinct and to the point explanations about countries using social factors derived from the World Value Survey. "
                    "You use the information given to you to answer questions about how countries score in various social factors that attempt to measure the social values held by the population of a country."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the World Value Survey for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        description = f"Here is a statistical description of the societal values of {self.country.name.capitalize()}."

        # subject_p, object_p, possessive_p = sentences.pronouns(country.gender)

        for metric in self.country.relevant_metrics:

            description += f"\n\nAccording to the WVS, {self.country.name.capitalize()} was found to "
            description += sentences.describe_level(
                self.country.ser_metrics[metric + "_Z"],
                thresholds=self.thresholds_dict[metric],
                words=self.description_dict[metric],
            )
            description += " compared to other countries in the same wave. "

            if metric in self.country.drill_down_metrics:

                if self.country.ser_metrics[metric + "_Z"] > 0:
                    index = 1
                else:
                    index = 0

                question, value = self.country.drill_down_metrics[metric]
                question, value = question[index], value[index]
                description += "In response to the question '"
                description += self.relevant_questions[metric][question][0]
                description += "', on average participants "
                description += self.relevant_questions[metric][question][1]
                description += " '"
                description += self.relevant_questions[metric][question][2][str(value)]
                description += "' "
                description += self.relevant_questions[metric][question][3]
                description += ". "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 2 short paragraph summary of the social values held by population of the country. "
            f"The first paragraph should focus on any factors or values for which the country is above or bellow average. If the country is neither above nor below average in any values, mention that. "
            f"The remaining paragraph should mention any specific values or factors that are neither high nor low compared to the average. "
        )
        return [{"role": "user", "content": prompt}]


class PersonDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward_bigfive.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward_bigfive.xlsx"]

    def __init__(self, person: Person):
        self.person = person
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a recruiter. "
                    "You provide succinct and to the point explanations about a candidate using data.  "
                    "You use the information given to you from the data and answers"
                    "to earlier user/assistant pairs to give summaries of candidates."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the candidate as a candidate or a person?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the candidate as a person. "
                    "When I say candidate, I mean person. "
                    "But I always talk about the candidate, as a person."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a candidate for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def categorie_description(self, value):
        if value <= -2:
            return "The candidate is extremely "
        elif -2 < value <= -1:
            return "The candidate is very "
        elif -1 < value <= -0.5:
            return "The candidate is quite "
        elif -0.5 < value <= 0.5:
            return "The candidate is relatively "
        elif 0.5 < value <= 1:
            return "The candidate is quite "
        elif 1 < value <= 2:
            return "The candidate is very "
        else:
            return "The candidate is extremely "

    def all_max_indices(self, row):
        max_value = row.max()
        return list(row[row == max_value].index)

    def all_min_indices(self, row):
        min_value = row.min()
        return list(row[row == min_value].index)

    def get_description(self, person):
        # here we need the dataset to check the min and max score of the person

        person_metrics = person.ser_metrics
        person_stat = PersonStat()
        questions = person_stat.get_questions()

        name = person.name
        extraversion = person_metrics["extraversion_Z"]
        neuroticism = person_metrics["neuroticism_Z"]
        agreeableness = person_metrics["agreeableness_Z"]
        conscientiousness = person_metrics["conscientiousness_Z"]
        openness = person_metrics["openness_Z"]

        text = []

        # extraversion
        cat_0 = "solitary and reserved. "
        cat_1 = "outgoing and energetic. "

        if extraversion > 0:
            text_t = (self.categorie_description(extraversion) 
            + cat_1
            + "The candidate tends to be more social. "
                     )
            if extraversion > 1:
                index_max = person_metrics[0:10].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (self.categorie_description(extraversion) 
            + cat_0
            + "The candidate tends to be less social. "
                     )
            if extraversion < -1:
                index_min = person_metrics[0:10].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # neuroticism
        cat_0 = "resilient and confident. "
        cat_1 = "sensitive and nervous. "

        if neuroticism > 0:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_1
                + "The candidate tends to feel more negative emotions and anxiety. "
            )
            if neuroticism > 1:
                index_max = person_metrics[10:20].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_0
                + "The candidate tends to feel less negative emotions and anxiety. "
            )
            if neuroticism < -1:
                index_min = person_metrics[10:20].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # agreeableness
        cat_0 = "critical and rational. "
        cat_1 = "friendly and compassionate. "

        if agreeableness > 0:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_1
                + "The candidate tends to be more cooperative, polite, kind and friendly. "
            )
            if agreeableness > 1:
                index_max = person_metrics[20:30].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_0
                + "The candidate tends to be less cooperative, polite, kind and friendly. "
            )
            if agreeableness < -1:
                index_min = person_metrics[20:30].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # conscientiousness
        cat_0 = "extravagant and careless. "
        cat_1 = "efficient and organized. "

        if conscientiousness > 0:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_1
                + "The candidate tends to be more careful or diligent. "
            )
            if conscientiousness > 1:
                index_max = person_metrics[30:40].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_0
                + "The candidate tends to be less careful or diligent. "
            )
            if conscientiousness < -1:
                index_min = person_metrics[30:40].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # openness
        cat_0 = "consistent and cautious. "
        cat_1 = "inventive and curious. "

        if openness > 0:
            text_t = (
                self.categorie_description(openness)
                + cat_1
                + "The candidate tends to be more open to new ideas and experiences. "
            )
            if openness > 1:
                index_max = person_metrics[40:50].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(openness)
                + cat_0
                + "The candidate tends to be less open to new ideas and experiences. "
            )
            if openness < -1:
                index_min = person_metrics[40:50].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        text = "".join(text)
        text = text.replace(",", "")
        return text

    def synthesize_text(self):
        person = self.person
        metrics = self.person.ser_metrics
        description = self.get_description(person)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the person's personality, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the person. "
            "The second sentence should describe the person's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the person is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the person compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]
