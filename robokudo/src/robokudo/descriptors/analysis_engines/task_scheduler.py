import py_trees

import robokudo.analysis_engine
import robokudo.pipeline
import robokudo.annotators.outputs

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.testing import SlowAnnotator, EmptyAnnotator

import robokudo.descriptors.camera_configs.config_kinect_robot

import robokudo.io.camera_interface

import robokudo.tree_components.task_scheduler


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self) -> str:
        return "job_scheduler"

    def implementation(self) -> robokudo.pipeline.Pipeline:
        """
        Create a pipeline which can schedule different perception tasks in a basic way
        """
        kinect_camera_config = (
            robokudo.descriptors.camera_configs.config_kinect_robot.CameraConfig()
        )
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(
                kinect_camera_config
            ),
        )

        # Annotator definition
        collection_reader = CollectionReaderAnnotator(descriptor=kinect_config)
        image_preprocessor = ImagePreprocessorAnnotator()
        slow1 = SlowAnnotator("Slow1")
        slow2 = SlowAnnotator("Slow2")

        # Subtree construction
        # A subtree is tailored towards solving a specific perception task
        #
        # Please note that the same instance of annotators might be used in multiple trees.
        # The scheduler/planning will take care of maintaining the correct relationships for dynamic trees.
        tree1 = py_trees.composites.Sequence("Tree1")
        tree2 = py_trees.composites.Sequence("Tree1")
        tree1.add_children(
            [collection_reader, image_preprocessor, slow1, slow2],
        )
        tree2.add_children(
            [collection_reader, image_preprocessor, slow1],
        )

        # Pipeline creation
        seq = robokudo.pipeline.Pipeline("Pipeline")
        # The Job Scheduler needs to be the first child of a Sequence
        task_scheduling = py_trees.composites.Sequence("Task Scheduling")
        task_scheduling.add_child(
            robokudo.tree_components.task_scheduler.IterativeTaskScheduler(
                tree_list=[tree1, tree2]
            )
        )

        seq.add_children(
            [
                robokudo.annotators.outputs.ClearAnnotatorOutputs(),
                EmptyAnnotator(),
                task_scheduling,
            ]
        )

        return seq
