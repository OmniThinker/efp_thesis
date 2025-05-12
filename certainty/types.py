from typing import TypedDict, Literal, List, Any

EventPolarity = Literal["Positive", "Negative"]
EventType = Literal['Attack',
                    'Transport',
                    'Die',
                    'Phone-Write',
                    'Injure',
                    'Meet',
                    'Transfer-Ownership',
                    'End-Position',
                    'Arrest-Jail',
                    'Demonstrate',
                    'Marry',
                    'Elect',
                    'Start-Position',
                    'Nominate',
                    'End-Org',
                    'Execute',
                    'Start-Org',
                    'Fine',
                    'Transfer-Money',
                    'Trial-Hearing',
                    'Sue',
                    'Charge-Indict',
                    'Sentence',
                    'Be-Born',
                    'Extradite',
                    'Declare-Bankruptcy',
                    'Convict',
                    'Release-Parole',
                    'Merge-Org',
                    'Appeal',
                    'Pardon',
                    'Divorce',
                    'Acquit']
EventGenericity = Literal["Specific", "Generic"]
EventModality = Literal["Asserted", "Other"]


class EventSentence(TypedDict):
    sent_id: str
    text: str
    type: EventType
    modality: EventModality
    polarity: EventPolarity
    genericity: EventGenericity
    trigger: str
    trigger_idx: Any
    label: int
    arguments: List[Any]
