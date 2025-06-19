import tensorflow as tf
import traceback
import logging

logger = logging.getLogger("Jaeger")


def configure_multi_gpu_inference(gpus):
    if gpus > 0:
        return tf.distribute.MirroredStrategy()
    else:
        None


def get_device_name(device):
    name = device.name
    return f"{name.split(':',1)[-1]}"


def create_virtual_gpus(logger, num_gpus=2, memory_limit=2048):
    if gpus := tf.config.list_physical_devices("GPU"):
        # Create n virtual GPUs with user defined amount of memory each
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)
                    for _ in range(num_gpus)
                ],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            logger.info(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logger.error(e)
            logger.debug(traceback.format_exc())
