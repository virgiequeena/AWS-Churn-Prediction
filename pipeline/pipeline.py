import sagemaker
import os
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# 1. Configuration
local_pipeline_session = LocalPipelineSession()

# In Learner Labs, it's safer to hardcode 'LabRole' if get_execution_role() fails
try:
    role = sagemaker.get_execution_role()
except:
    role = 'LabRole'

instance_type = "local"

base_path = "/home/ec2-user/SageMaker"

# 1. Use the CLEAN path for creating folders
os.makedirs(f"{base_path}/ingested", exist_ok=True)
os.makedirs(f"{base_path}/train", exist_ok=True)
os.makedirs(f"{base_path}/test", exist_ok=True)
os.makedirs(f"{base_path}/eval", exist_ok=True)

# 2. Use the URI (file://) ONLY for SageMaker Step definitions
local_output_path   = "file:///home/ec2-user/SageMaker"
local_ingested_data = f"{local_output_path}/ingested"
local_train_data    = f"{local_output_path}/train"
local_test_data     = f"{local_output_path}/test"
local_eval_data     = f"{local_output_path}/eval"


# 3. Processor & Estimator
sklearn_processor = SKLearnProcessor(
    framework_version='1.2-1',
    role=role,
    instance_type=instance_type,
    instance_count=1,
    sagemaker_session=local_pipeline_session
)

sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type=instance_type,
    framework_version='1.2-1',
    sagemaker_session=local_pipeline_session,
    hyperparameters={"max_depth": 4}
)

# 3. Steps

# --- STEP 1: INGESTION ---
step_ingest = ProcessingStep(
    name="ChurnIngest",
    processor=sklearn_processor, 
    inputs=[
        ProcessingInput(
            source="/home/ec2-user/SageMaker",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="ingested_data", 
            source="/opt/ml/processing/ingested",
            destination=local_ingested_data
        )
    ],
    code="data_ingestion.py"
)

# --- STEP 2: PREPROCESS ---
step_preprocess = ProcessingStep(
    name="ChurnPreprocess",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=step_ingest.properties.ProcessingOutputConfig.Outputs["ingested_data"].S3Output.S3Uri, 
            destination="/opt/ml/processing/ingested"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train", 
            source="/opt/ml/processing/train", 
            destination=local_train_data
        ),
        ProcessingOutput(
            output_name="test", 
            source="/opt/ml/processing/test", 
            destination=local_test_data
        )
    ],
    code="preprocessing.py"
)

# --- STEP 3: TRAINING ---
step_train = TrainingStep(
    name="ChurnTrain",
    estimator=sklearn_estimator,
    inputs={"train": TrainingInput(s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri)}
)

# --- STEP 4: EVALUATION ---
evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

step_eval = ProcessingStep(
    name="ChurnEval",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )],

    code="evaluation.py",
    property_files=[evaluation_report]
)

pipeline = Pipeline(
    name="Churn-Local-Workflow",
    steps=[step_ingest, step_preprocess, step_train, step_eval],
    sagemaker_session=local_pipeline_session
)

# 4. Run
pipeline.upsert(role_arn=role)
execution = pipeline.start()
