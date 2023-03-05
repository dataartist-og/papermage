"""

Annotations are objects that are 'aware' of the Document. For example, imagine an entity
in a document; representing it as an Annotation data type would allow you to access the
Document object directly from within the Entity itself.

@kylel

"""

import logging

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Union, Optional

if TYPE_CHECKING:
    # from mmda.types.document import Document
    pass


class Annotation:
    """Represent a "unit" (e.g. highlighted span, drawn boxes) layered on a Document."""

    @abstractmethod
    def __init__(self):
        self._id: Optional[int] = None
        self._doc: Optional['Document'] = None
        # logging.warning('Unless testing or developing, we dont recommend creating Annotations '
        #                 'manually. Annotations need to store things like `id` and references '
        #                 'to a `Document` to be valuable. These are all handled automatically in '
        #                 '`Parsers` and `Predictors`.')

    @property
    def doc(self) -> Optional['Document']:
        return self._doc

    @doc.setter
    def doc(self, doc: 'Document') -> None:
        """This method attaches a Document to this Annotation, allowing the Annotation
        to access things beyond itself within the Document (e.g. neighboring annotations)"""
        if self.doc:
            raise AttributeError("Already has an attached Document. Since Annotations should be"
                                 "specific to a given Document, we recommend creating a new"
                                 "Annotation from scratch and then attaching your Document.")
        self._doc = doc

    @property
    def id(self) -> Optional[int]:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        """This method assigns an ID to an Annotation. Requires a Document to be attached
        to this Annotation. ID basically gives the Annotation itself awareness of its
        position within the broader Document."""
        if self.id:
            raise AttributeError(f"This Annotation already has an ID: {self.id}")
        if not self.doc:
            raise AttributeError('This Annotation is missing a Document')
        self._id = id

    @abstractmethod
    def to_json(self) -> Union[Dict, List]:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, annotation_json: Union[Dict, List]) -> "Annotation":
        pass
