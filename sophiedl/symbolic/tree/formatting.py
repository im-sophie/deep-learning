def format_newline(indent: int, indent_width: int) -> str:
    assert indent >= 0
    assert indent_width > 0
    return "\n{0}".format(
        " " * (indent * indent_width)
    )

def format_color_default(color: bool) -> str:
    if color:
        return "\033[0m"
    else:
        return ""

def format_color_type(color: bool) -> str:
    if color:
        return "\033[1;36m"
    else:
        return ""

def format_color_symbol(color: bool) -> str:
    if color:
        return "\033[1;33m"
    else:
        return ""

def format_color_literal(color: bool) -> str:
    if color:
        return "\033[0;35m"
    else:
        return ""

def format_color_member_name(color: bool) -> str:
    if color:
        return "\033[2;32m"
    else:
        return ""
