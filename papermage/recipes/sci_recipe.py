"""

@kylel

"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union
import yaml

from papermage.magelib import (
    AbstractsFieldName,
    AlgorithmsFieldName,
    AuthorsFieldName,
    BibliographiesFieldName,
    BlocksFieldName,
    Box,
    CaptionsFieldName,
    Document,
    EntitiesFieldName,
    Entity,
    EquationsFieldName,
    FiguresFieldName,
    FootersFieldName,
    FootnotesFieldName,
    HeadersFieldName,
    ImagesFieldName,
    KeywordsFieldName,
    ListsFieldName,
    LatexFieldName,
    InlineLatexFieldName,
    VoidFieldName,
    PagesFieldName,
    ParagraphsFieldName,
    RelationsFieldName,
    RowsFieldName,
    SectionsFieldName,
    SentencesFieldName,
    SymbolsFieldName,
    TablesFieldName,
    TitlesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import (
    HFBIOTaggerPredictor,
    IVILATokenClassificationPredictor,
    LPEffDetFormulaPredictor,
    LPEffDetPubLayNetBlockPredictor,
    PysbdSentencePredictor,
    SVMWordPredictor,
    MathPredictor,
    LayoutLMv3Predictor
)
from papermage.predictors.word_predictors import make_text
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer
from papermage.recipes.recipe import Recipe
from papermage.utils.annotate import group_by

VILA_LABELS_MAP = {
    "Title": TitlesFieldName,
    "Paragraph": ParagraphsFieldName,
    "Author": AuthorsFieldName,
    "Abstract": AbstractsFieldName,
    "Keywords": KeywordsFieldName,
    "Section": SectionsFieldName,
    "List": ListsFieldName,
    "Bibliography": BibliographiesFieldName,
    "Equation": EquationsFieldName,
    "Algorithm": AlgorithmsFieldName,
    "Figure": FiguresFieldName,
    "Table": TablesFieldName,
    "Caption": CaptionsFieldName,
    "Header": HeadersFieldName,
    "Footer": FootersFieldName,
    "Footnote": FootnotesFieldName,
}

LAYOUTLMV3_LABELS_MAP = {
    "title": TitlesFieldName,
    "plain_text": ParagraphsFieldName,
    "abandon": VoidFieldName,
    "figure": FiguresFieldName,
    "figure_caption": CaptionsFieldName,
    "table": TablesFieldName,
    "table_caption": CaptionsFieldName,
    "table_footnote": FootnotesFieldName,
    "isolate_formula": LatexFieldName,
    "formula_caption": CaptionsFieldName,
    "inline_formula": InlineLatexFieldName,
    "isolated_formula": EquationsFieldName
}

class SciRecipe(Recipe):
    def __init__(
        self,
        ivila_predictor_path: str = "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
        bio_roberta_predictor_path: str = "allenai/vila-roberta-large-s2vl-internal",
        svm_word_predictor_path: str = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/mmda/models/svm_word_predictor.tar.gz",
        # layoutlmv3_predictor_path: str = "microsoft/layoutlmv3-base",
        dpi: int = 200,
        mf_config_path: str = 'configs/model_configs.yaml'
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dpi = dpi

        self.logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.word_predictor = SVMWordPredictor.from_path(svm_word_predictor_path)

        self.publaynet_block_predictor = LPEffDetPubLayNetBlockPredictor.from_pretrained()
        self.ivila_predictor = IVILATokenClassificationPredictor.from_pretrained(ivila_predictor_path)
        self.bio_roberta_predictor = HFBIOTaggerPredictor.from_pretrained(
            bio_roberta_predictor_path,
            entity_name="tokens",
            context_name="pages",
        )
        self.sent_predictor = PysbdSentencePredictor()
        self.math_predictor = MathPredictor(mf_config_path, verbose=True)
        with open('configs/model_configs.yaml') as f:
            model_configs = yaml.load(f, Loader=yaml.FullLoader)
        self.layoutlmv3_predictor = LayoutLMv3Predictor.from_pretrained(weight = model_configs['model_args']['layout_weight'], device = model_configs['model_args']['device'])
        self.logger.info("Finished instantiating recipe")

    def from_pdf(self, pdf: Path) -> Document:
        self.logger.info("Parsing document...")
        doc, is_blank = self.parser.parse(input_pdf_path=pdf, return_blank_page_mask=True)

        self.logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdf, dpi=self.dpi)
        images = [im for im, blank in zip(images, is_blank) if not blank]
        
        doc.annotate_images(images=list(images))
        self.rasterizer.attach_images(images=images, doc=doc)
        return self.from_doc(doc=doc)

    def from_doc(self, doc: Document) -> Document:
        self.logger.info("Predicting words...")
        words = self.word_predictor.predict(doc=doc)
        doc.annotate_layer(name=WordsFieldName, entities=words)

        self.logger.info("Predicting sentences...")
        sentences = self.sent_predictor.predict(doc=doc)
        doc.annotate_layer(name=SentencesFieldName, entities=sentences)

        self.logger.info("Predicting blocks...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = self.publaynet_block_predictor.predict(doc=doc)
        doc.annotate_layer(name=BlocksFieldName, entities=blocks)

        self.logger.info("Predicting vila...")
        vila_entities = self.ivila_predictor.predict(doc=doc)
        doc.annotate_layer(name="vila_entities", entities=vila_entities)

        for entity in vila_entities:
            entity.boxes = [
                Box.create_enclosing_box(
                    [b for t in doc.intersect_by_span(entity, name=TokensFieldName) for b in t.boxes]
                )
            ]
            entity.text = make_text(entity=entity, document=doc)
        preds = group_by(entities=vila_entities, metadata_field="label", metadata_values_map=VILA_LABELS_MAP)
        doc.annotate(*preds)

        self.logger.info("Predicting latex formulas...")
        latex_entities = self.math_predictor.predict(doc=doc)
        doc.annotate_layer(name=LatexFieldName, entities=latex_entities)

        for entity in latex_entities:
            entity.boxes = [
                Box.create_enclosing_box(
                    [b for t in doc.intersect_by_span(entity, name=TokensFieldName) for b in t.boxes]
                )
            ]
        for entity in sorted(latex_entities, key=lambda e: (e.end, e.start), reverse=True):
            doc.symbols[entity.start : entity.end] = entity.metadata.latex

        self.logger.info("Predicting LayoutLMv3...")
        layoutlmv3_entities = self.layoutlmv3_predictor.predict(doc=doc)
        doc.annotate_layer(name="layoutlmv3_entities", entities=layoutlmv3_entities)

        # for entity in layoutlmv3_entities:
        #     entity.text = make_text(entity=entity, document=doc)
        layoutlmv3_preds = group_by(entities=layoutlmv3_entities, metadata_field="label", metadata_values_map=LAYOUTLMV3_LABELS_MAP)
        doc.annotate(*layoutlmv3_preds)

        return doc


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file.")
    parser.add_argument("--output", type=str, help="Path to output JSON file.")
    args = parser.parse_args()

    recipe = SciRecipe()
    doc = recipe.from_pdf(pdf=args.pdf)
    with open(args.output, "w") as f:
        json.dump(doc.to_json(), f, indent=2)
