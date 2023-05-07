import sys
from pose_utils import gen_poses


import configargparse
parser = configargparse.ArgumentParser()
parser.add_argument('--match_type', type=str, 
					default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
# 实际上是`path/to/project/`, your images are located in `path/to/project/images`
parser.add_argument('-d', '--data_dir', type=str, help='input scene directory')
parser.add_argument('-f', '--factor', type=int, default=4, help='minify factor, downsample factor for images')
args = parser.parse_args()

if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
	print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
	sys.exit()

if __name__=='__main__':
    gen_poses(args.data_dir, args.match_type, [args.factor])