"""Metadata-only traversal of current and historical user audio."""


def user_audio_turns(item, *, strict_roles=False):
    conversation_id = item["id"]
    yield True, conversation_id, None, item.get("audio")
    for index, turn in enumerate(item.get("history", [])):
        role = turn.get("role")
        if role == "user":
            yield False, f"{conversation_id}_history_{index:03d}", index, turn.get("audio")
        elif strict_roles and role != "assistant":
            raise ValueError(f"Invalid history role: {conversation_id}/{index}")
