
from Referto import Referto
from datetime import datetime
from typing import Protocol, List


class IRefertationRepository(Protocol):

    def get_by_id(self, referto_id: int) -> Referto | None:
        ... 
    def get_list_by_paziente(self, paziente_id: int) -> List[Referto]:
        ...
    def get_list_by_paziente_date(self, paziente_id: int,start:datetime,end:datetime) -> List[Referto]:
        ...
    