"""
Shared utility helpers used across Streamlit pages and plotting/chat flows.

This module mainly provides:
- string normalization/formatting helpers for labels and generated text,
- export/render helpers for DataFrames and matplotlib figures,
- selection helpers that convert filtered datasets into data-point objects, and
- chat construction helper based on a deterministic in-session hash key.
"""

from io import BytesIO
import copy

import matplotlib.colors as c


def split_names(player_names):
    """
    Return simplified display names derived from full player names.

    Rules:
    - For single-token names, return that token.
    - For most multi-token names, return only the final token.
    - If the second-to-last token has length 2, keep the final 2 tokens together (for example surname particles like "de", "da", "Al").

    Args:
        player_names: Iterable of full player-name strings.

    Returns:
        list[str]: Simplified names suitable for compact labels.
    """
    return [
        # Keep the final token by default.
        (
            name.split()[-1]
            if len(name.split()) == 1 or len(name.split()[-2]) != 2
            # Preserve two-part surname endings when particle length is 2.
            else " ".join(name.split()[-2:])
        )
        for name in player_names
    ]


def add_per_90(attributes):
    """
    Append ``" per 90"`` to raw attribute labels when appropriate.

    This is a lightweight display helper used for chart/table labels.

    Args:
        attributes: Iterable of metric/attribute names.

    Returns:
        list[str]: Updated attribute names.

    Notes:
        The suffix is not added when a label appears percentage-like or already rate/efficiency-adjusted (contains `%`, `per`, `adj`, `eff`, or ` - `).
    """
    return [
        (
            c + " per 90"
            if "%" not in c
            and "per" not in c
            and "adj" not in c
            and "eff" not in c
            and " - " not in c
            else c
        )
        for c in attributes
    ]


def normalize_text(s, sep_token=" \n "):
    """
    Normalize text spacing/punctuation artifacts before downstream use.

    Args:
        s: Raw text input.
        sep_token: Legacy compatibility argument (currently unused).

    Returns:
        str: Cleaned text with compact spacing and corrected punctuation joins.
    """
    # Collapse repeated whitespace first, then apply targeted punctuation fixes.
    s = " ".join(s.split())
    s = s.replace(". ,", ",")
    s = s.replace(" ,", ",")
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s


def insert_newline(s, n_length=15):
    """
    Insert a newline near a target width if a split point exists.

    The function searches for the last space before ``n_length`` and inserts a newline there. If no suitable space exists, the original string is returned.

    Args:
        s: Input string.
        n_length: Preferred maximum length before line break.

    Returns:
        str: Possibly line-broken string.
    """
    if len(s) <= n_length:
        return s
    else:
        last_space_before_15 = s.rfind(" ", 0, n_length)
        if last_space_before_15 == -1:  # No space found within the first 15 characters
            return s  # Return original string
        else:
            # Split the string at the space and insert a newline
            return s[:last_space_before_15] + "\n" + s[last_space_before_15 + 1 :]


def rgba_to_hex(rgba):
    """
    Convert an RGBA tuple of 0-1 floats to a hex RGB color string.

    Args:
        rgba: Tuple-like ``(r, g, b, a)`` where channels are in ``[0, 1]``.

    Returns:
        str: Hex color in ``#rrggbb`` format.

    Notes:
        Alpha is accepted for API compatibility but not encoded in output.
    """
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def convert_df_to_csv(df, n=1000, ignore=[]):
    """
    Serialize the first ``n`` rows of a DataFrame as UTF-8 CSV bytes.

    Args:
        df: Source pandas DataFrame.
        n: Maximum number of rows to include (head subset).
        ignore: Legacy compatibility argument (currently unused).

    Returns:
        bytes: Encoded CSV payload suitable for Streamlit download widgets.
    """
    return df.head(n).to_csv(index=None).encode("utf-8")


def get_img_bytes(fig, custom=False, format="png", dpi=200):
    """
    Render a matplotlib figure into an in-memory byte buffer.

    Args:
        fig: Matplotlib figure object implementing ``savefig``.
        custom: When ``True``, use tight bounding box and extra padding.
        format: Output image format passed to ``savefig``.
        dpi: Render DPI passed to ``savefig``.

    Returns:
        io.BytesIO: Rewound buffer ready for direct read/download.
    """
    tmpfile = BytesIO()

    if custom:
        # Tight export is useful for share/download assets with cleaner framing.
        fig.savefig(
            tmpfile,
            format=format,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.35,
        )
    else:
        fig.savefig(
            tmpfile,
            format=format,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            transparent=False,
        )  # , frameon=False)  # , transparent=False, bbox_inches='tight', pad_inches=0.35)

    # Rewind so consumers can read from the beginning of the buffer.
    tmpfile.seek(0)

    return tmpfile


def hex_color_transparency(hex, alpha):
    """
    Apply an alpha value to a hex color and return RGBA-hex string.

    Args:
        hex: Base color string accepted by matplotlib (typically ``#rrggbb``).
        alpha: Transparency in ``[0, 1]``.

    Returns:
        str: Hex string including alpha channel (``#rrggbbaa``).
    """
    return c.to_hex(c.to_rgba(hex, alpha), True)


def select_player(container, players, gender, position):
    """
    Select one player in UI and return it as a data-point object.

    Args:
        container: Streamlit container/context where selection widgets render.
        players: Player collection object supporting ``select_and_filter`` and
            ``to_data_point``.
        gender: Gender label passed through to ``to_data_point``.
        position: Position label passed through to ``to_data_point``.

    Returns:
        Data-point representation of the selected player.
    """

    # Work on a copy so interactive filtering does not mutate shared state.
    player = copy.deepcopy(players)

    # Render selection widgets inside the provided Streamlit container.
    with container:

        # Standard player-name selector expected by downstream pages.
        player.select_and_filter(
            column_name="player_name",
            label="Player",
        )

        # Convert filtered row to domain data-point object.
        player = player.to_data_point(gender, position)

    return player


def select_country(container, countries):
    """
    Select one country in UI and return it as a data-point object.

    Args:
        container: Streamlit container/context where selection widgets render.
        countries: Country collection object supporting ``select_and_filter``
            and ``to_data_point``.

    Returns:
        Data-point representation of the selected country.
    """

    # Work on a copy so interactive filtering does not mutate shared state.
    country = copy.deepcopy(countries)

    # rnd = int(country.select_random()) # does not work because of page refresh!
    # Render selection widgets inside the provided Streamlit container.
    with container:

        # Standard country selector expected by WVS page flows.
        country.select_and_filter(
            column_name="country",
            label="Country",
            # default_index=rnd,  # randomly select a country for default
        )

        # Convert filtered row to domain data-point object.
        country = country.to_data_point()

    return country


def create_chat(to_hash, chat_class, *args, **kwargs):
    """
    Instantiate a chat object keyed by a hash of session-defining inputs.

    Args:
        to_hash: Any hashable object/tuple describing chat identity context
            (for example selected player id + page name).
        chat_class: Chat class to instantiate. Its constructor must accept the
            hash state as the first positional argument.
        *args: Forwarded to ``chat_class`` constructor.
        **kwargs: Forwarded to ``chat_class`` constructor.

    Returns:
        Chat instance initialized with computed hash state.
    """
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat
