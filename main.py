import argparse
from image_segmentation import image_seg_clustering, save_images

def main(args):
    original, segmented = image_seg_clustering(args.img_path, args.k, args.max_iters)
    save_images(original, segmented, args.original_output, args.segmented_output)
    print(f"Original image saved to {args.original_output}")
    print(f"Segmented image saved to {args.segmented_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Segmentation using KMeans")
    parser.add_argument("img_path", type=str, help="Path to the input image")
    parser.add_argument("original_output", type=str, help="Path to save the original image")
    parser.add_argument("segmented_output", type=str, help="Path to save the segmented image")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters for KMeans")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum iterations for KMeans")
    
    args = parser.parse_args()
    main(args)