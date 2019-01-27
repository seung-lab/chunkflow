import click
import ast


# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.group()
@click.option(
    '--output_chunk_start',
    type=int,
    help="the start coordinates of final output block",
    cls=PythonLiteralOption,
    default=[0, 0, 0])
@click.option(
    '--output_chunk_size',
    type=int,
    help="the size of output block",
    cls=PythonLiteralOption,
    default=[112, 1152, 1152])
@click.option(
    '--overlap',
    type=int,
    help="overlap by number of voxels",
    cls=PythonLiteralOption,
    default=[4, 64, 64])
@click.option(
    '--patch_size',
    type=int,
    help="convnet input patch size",
    cls=PythonLiteralOption,
    default=[32, 256, 256])
# @click.option('--no_eval', action='store_true', help="this is on then using dynamic \ batchnorm, otherwise static.")
# @click.option('--output_key', type=str, default='affinity', help="the name of the final output layer")
def main():
    """
    Set up configuration
    """
    print('hi')


@main.command()
@click.option(
    '--input_image_source',
    type=str,
    help="input image source path, i.e. file://, gs://, or s3://.",
    required=True)
@click.option(
    '--output_core_destination',
    type=str,
    help="destination path for the valid core output of the chunk,\
              prefixes supported: file://, gs://, s3://.",
    required=True)
@click.option(
    '--output_overlap_destination',
    type=str,
    help="destination path of the overlap region of the chunk,\
              prefixes supported: file://, gs://, s3://.",
    required=False)
@click.option(
    '--model_path', type=str, help="the path of convnet model", required=True)
@click.option(
    '--net_path', type=str, help="the path of convnet weights", required=True)
@click.option(
    '--gpu_ids',
    type=int,
    help="ids of gpus to use",
    cls=PythonLiteralOption,
    default=None)
@click.option(
    '--framework',
    type=str,
    help="backend of deep learning framework, such as pytorch and pznet.",
    default='pytorch')
@click.option(
    '--output_channels',
    type=int,
    help="number of convnet output channels",
    default=3)
def inference():
    """
    Run inference on a block
    """
    from chunkflow.block_inference_engine import BlockInferenceEngine
    from chunkflow.frameworks.pznet_patch_inference_engine import PZNetPatchInferenceEngine
    from chunkflow.frameworks.pytorch_patch_inference_engine import PytorchPatchInferenceEngine
    if args.backend == "pytorch":
        patch_engine = PytorchPatchInferenceEngine(model_path, net_path)
    elif args.backend == "pznet":
        # the net path is a directory which contains both model and weights
        patch_engine = PZNetPatchInferenceEngine(net_path)
    else:
        raise "invalid inference backend, only support pytorch and pznet for now."

    block_inference_engine = BlockInferenceEngine(
        patch_inference_engine=patch_engine,
        patch_size=tuple(p for p in args.patchsize),
        overlap=tuple(o for o in args.patchoverlap),
        output_key=args.outputlayername,
        output_channels=args.outputchannelnum)

    # img = np.asarray(img, dtype='float32') / 255.0
    img = OffsetArray(img)
    output = block_inference_engine(img)

    print('shape of output: {}'.format(output.shape))
    imsave(output, args.outputfile)


@click.option(
    '--input_overlap_sources',
    type=str,
    help="input source path(s) for overlap regions to blend \
              prefixes supported: file://, gs://, s3://. i.e. [\"gs://mybucket/left\", \"gs://mybucket/right\"]",
    required=True)
@click.option(
    '--output_final_destination',
    type=str,
    help="final destination path of chunkflow blended inference,\
              prefixes supported: file://, gs://, s3://.",
    required=True)
@main.command()
def blend():
    """
    Blend chunk using overlap regions
    """
    print('blend')


if __name__ == '__main__':
    main()
