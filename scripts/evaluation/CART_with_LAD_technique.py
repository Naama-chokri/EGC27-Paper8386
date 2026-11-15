from scripts.utility.change_directory_and_delete_file import \
    change_directory_and_delete_all_input_and_output_hypergraph_files_for_binary_dataset
# without feature selection
from scripts.evaluation.evaluation_CART_without_FS import \
    evaluation_CART_without_FS
# LAD-based methods
from scripts.evaluation.evaluation_CART_with_LAD_based_FS import evaluation_CART_with_LAD_based_FS


def write_single_result_for_CART_using_LAD_based_technique(dataset_name, max_depth, k_fold, n_repeats, seed):
    avg_acc_cart, avg_time_cart = evaluation_CART_without_FS(dataset_name=dataset_name,
                                                             max_depth=max_depth,
                                                             k_fold=k_fold, n_repeats=n_repeats,
                                                             seed=seed)

    avg_acc_lad, list_smallest_subset_lad, avg_length_subset_lad, avg_MHSes_time_lad, avg_time_subsets_ranking, avg_time_lad = evaluation_CART_with_LAD_based_FS(
        dataset_name=dataset_name,
        max_depth=max_depth,
        k_fold=k_fold,
        n_repeats=n_repeats,
        seed=seed)

    write_output_results_path = "output_pmmcs/" + dataset_name + "/max_depth_" + str(
        max_depth) + "/results_cart_depth_" + str(max_depth) + "_after_" + str(
        k_fold) + "_folds_and_" + str(n_repeats) + "_repeats.txt"
    with open(write_output_results_path, "w") as output_results:
        output_results.write("######################################################################\n"
                             "#                      CART summary results                          #\n"
                             "######################################################################\n")
        output_results.write("avg_test_accuracy: " + str(avg_acc_cart) + "\n")
        output_results.write("avg_time: " + str(avg_time_cart) + "\n")
        output_results.write("######################################################################\n"
                             f"# LAD-based + CART for smallest combinations of features            #\n"
                             "######################################################################\n")

        output_results.write("avg_test_accuracy: " + str(avg_acc_lad) + "\n")
        output_results.write("smallest subset found: " + str(list_smallest_subset_lad) + "\n")
        output_results.write("avg length subset: " + str(avg_length_subset_lad) + "\n")
        output_results.write("avg time for generating 10000 MHSes: " + str(avg_MHSes_time_lad) + "\n")
        output_results.write("avg time for ranking subsets: " + str(avg_time_subsets_ranking) + "\n")
        output_results.write("avg time for lad: " + str(avg_time_lad) + "\n")
        # delete all temporary files of LAD-based method
        change_directory_and_delete_all_input_and_output_hypergraph_files_for_binary_dataset(
            dataset_name=dataset_name,
            max_depth=max_depth, depth=-1)

        print("Finished writing results...\n\n\n")
    print(f"After {k_fold} cross validations with {n_repeats} repetitions:\n")
    print("######################################################################\n"
          "#                      CART summary results                          #\n"
          "######################################################################")
    print("avg_test_accuracy: " + str(avg_acc_cart))
    print("avg_time: " + str(avg_time_cart)+ "\n")
    print("######################################################################\n"
         f"# LAD-based + CART for smallest combinations of features             #\n"
          "######################################################################")
    print("avg_test_accuracy: " + str(avg_acc_lad))
    print("smallest subset found: " + str(list_smallest_subset_lad))
    print("avg length subset: " + str(avg_length_subset_lad))
    print("avg time for generating 10000 MHSes: " + str(avg_MHSes_time_lad))
    print("avg time for ranking subsets: " + str(avg_time_subsets_ranking))
    print("avg time for lad: " + str(avg_time_lad))
