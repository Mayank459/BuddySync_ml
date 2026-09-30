from fastapi import APIRouter
from pydantic import BaseModel

from buddy_matching.api import artifact, tables
from clan_formation.clans import form_clans, pair_matrix

router = APIRouter(prefix="/clans", tags=["clan formation"])


class ClanQuery(BaseModel):
    event_id: int
    size: int = 4
    min_score: float = 0.4


@router.post("/form")
def form(q: ClanQuery):
    """Batch job endpoint: propose clans among people at this event who want a group."""
    users, participants, _ = tables()
    wants_group = set(users.loc[users.group_pref != "duo", "user_id"])
    ids = participants.loc[(participants.event_id == q.event_id)
                           & (participants.status == "looking_for_buddy")
                           & participants.user_id.isin(wants_group), "user_id"].tolist()
    if len(ids) < 3:
        return {"event_id": q.event_id, "clans": []}
    M = pair_matrix(ids, users, artifact())
    clans = form_clans(M, k=q.size, tau=q.min_score)
    return {"event_id": q.event_id,
            "clans": [{"members": [int(ids[m]) for m in c],
                       "worst_pair_pct": int(round(100 * min(M[a, b] for x, a in enumerate(c) for b in c[x + 1:])))}
                      for c in clans]}
