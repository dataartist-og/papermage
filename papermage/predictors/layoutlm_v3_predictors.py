from typing import List, Optional, Dict
from tqdm import tqdm
import numpy as np
from papermage.magelib import (
    Box,
    Document,
    Entity,
    Image,
    ImagesFieldName,
    Metadata,
    PagesFieldName,
)
from papermage.predictors import BasePredictor
from papermage.modules.layoutlmv3.model_init import Layoutlmv3_Predictor

class LayoutLMv3Predictor(BasePredictor):
    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [PagesFieldName, ImagesFieldName]

    def __init__(self, model):
        self.model = model
        self.id2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption", "table_footnote", 
                         "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula", "ocr_text"]

    @classmethod
    def from_pretrained(
        cls,
        weight: str,
        device: Optional[str] = 'cuda:0',
    ):
        model = Layoutlmv3_Predictor(weight)
        return cls(model)

    def postprocess(self, layout_result: Dict, page_index: int, image: np.ndarray) -> List[Entity]:
        page_height, page_width = image.shape[0], image.shape[1]
        entities = []

        for layout_det in layout_result["layout_dets"]:
            category_id = layout_det["category_id"]
            poly = layout_det["poly"]
            score = layout_det["score"]

            x_min, y_min = poly[0], poly[1]
            x_max, y_max = poly[4], poly[5]

            box = Box(
                l=x_min,
                t=y_min,
                w=x_max - x_min,
                h=y_max - y_min,
                page=page_index,
            ).to_relative(
                page_width=page_width,
                page_height=page_height,
            )

            entity = Entity(
                boxes=[box],
                metadata=Metadata(
                    label=self.id2names[category_id],
                    score=score
                )
            )
            entities.append(entity)

        return entities

    def _predict(self, doc: Document) -> List[Entity]:
        document_prediction = []

        images = doc.get_layer(name=ImagesFieldName)
        for image_index, pm_image in enumerate(tqdm(images)):
            image = pm_image.to_array()
            layout_result = self.model(image)
            document_prediction.extend(self.postprocess(layout_result, image_index, image))
		
        return document_prediction
