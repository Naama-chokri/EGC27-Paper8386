import sys
from scripts.evaluation.CART_with_LAD_technique import \
    write_single_result_for_CART_using_LAD_based_technique

dataset_name = sys.argv[1]
max_depth = int(sys.argv[2])

print(f"Using dataset: '{dataset_name}'")
print(f"Using max_depth: '{max_depth}'")

write_single_result_for_CART_using_LAD_based_technique(dataset_name=dataset_name, max_depth=max_depth,
                                                       k_fold=10, n_repeats=3, seed=2024)
