#!/usr/bin/python 
import argparse 
from chunkflow.offset_array import OffsetArray

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile",
                        default = "/tmp/image.h5",
                        help = "input file path")
    parser.add_argument("-o", "--outputfile",
                        default = "/tmp/output.h5",
                        help = "output file path")
    parser.add_argument("-b", "--backend",
                        default = "pytorch",
                        help="inference engine in the backend: {pytorch, pznet}.")
    parser.add_argument("-m", "--modelfile",
                        help="convnet model file path")
    parser.add_argument("-w", "--weightfile",
                        help = "convnet directory path")
    parser.add_argument("-p", "--patchsize",
                        type = int, nargs = "+",
                        help = "input patch size")
    parser.add_argument("-v", "--patchoverlap",
                        type = int, nargs="+",
                        help = "overlap of neighboring patches")
    parser.add_argument("-l", "--outputlayername",
                        help = "the layer name of final output layer")
    parser.add_argument("-c", "--outputchannelnum",
                        type = int, default = 3,
                        help = "number of output channel")
    args = parser.parse_args()

    from chunkflow.block_inference_engine import BlockInferenceEngine
    from chunkflow.frameworks.pznet_patch_inference_engine import PZNetPatchInferenceEngine
    if args.backend == "pytorch":
        patch_engine = PytorchPatchInferenceEngine( args.modelfile, args.weightfile )
    elif args.backend == "pznet":
        patch_engine = PZNetPatchInferenceEngine( args.modelfile )
    else:
        raise "invalid inference backend, only support pytorch and pznet for now."

    block_inference_engine = BlockInferenceEngine(
        patch_inference_engine=patch_engine,
        patch_size = tuple( p for p in args.patchsize ),
        overlap = tuple(o for o in args.patchoverlap),
        output_key = args.outputlayername,
        output_channels = args.outputchannelnum)

    from emirt.emio import imread, imsave
    img = imread( args.inputfile )
    # img = np.asarray(img, dtype='float32') / 255.0
    img = OffsetArray(img)
    output = block_inference_engine(img)

    print('shape of output: {}'.format(output.shape))
    imsave( output, args.outputfile )

main()
