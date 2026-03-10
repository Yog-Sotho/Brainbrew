from config import DistillationConfig, QualityMode

from pipeline.document_loader import load_document, chunk_text
from pipeline.evol_instruct import evolve
from pipeline.dataset_builder import build_with_vllm
from pipeline.cleaners import clean_dataset
from pipeline.evaluator import evaluate_dataset
from pipeline.exporter import export_alpaca

from engines.teacher_engine import TeacherEngine
from engines.vllm_engine import VLLMEngine
from engines.judge_engine import JudgeEngine
from engines.ensemble_engine import EnsembleTeacher

from training.lora_trainer import train_lora
from publish.hf_publisher import publish_dataset


def create_teacher(cfg):

    models = [m.strip() for m in cfg.teacher_model.split(",")]

    engines = []

    for m in models:

        if cfg.use_vllm:
            engines.append(VLLMEngine(m))
        else:
            engines.append(TeacherEngine(m))

    if len(engines) == 1:
        return engines[0]

    return EnsembleTeacher(engines)


def run_distillation(cfg: DistillationConfig):

    text = load_document(cfg.source_file)

    chunks = chunk_text(text)

    prompts = [
        f"Explain the following concept clearly:\n\n{c}"
        for c in chunks
    ]

    if cfg.quality_mode == QualityMode.RESEARCH:
        prompts = evolve(prompts,3)

    elif cfg.quality_mode == QualityMode.BALANCED:
        prompts = evolve(prompts,2)

    prompts = prompts[:cfg.dataset_size]

    teacher = create_teacher(cfg)

    build_with_vllm(
        prompts,
        teacher,
        cfg.raw_dataset,
        batch_size=64
    )

    clean_dataset(cfg.raw_dataset, cfg.clean_dataset)

    if cfg.quality_mode != QualityMode.FAST:

        judge_engine = TeacherEngine(cfg.judge_model)

        judge = JudgeEngine(judge_engine)

        threshold = 6 if cfg.quality_mode == QualityMode.BALANCED else 8

        evaluate_dataset(
            cfg.clean_dataset,
            cfg.scored_dataset,
            judge,
            threshold
        )

        export_alpaca(cfg.scored_dataset, cfg.final_dataset)

    else:

        export_alpaca(cfg.clean_dataset, cfg.final_dataset)

    if cfg.train_model:

        train_lora(
            cfg.final_dataset,
            cfg.base_model,
            "trained_model"
        )

    if cfg.publish_dataset:

        publish_dataset(
            cfg.final_dataset,
            cfg.hf_repo
        )

    return cfg.final_dataset
