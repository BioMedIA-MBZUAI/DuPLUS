
def get_dataset(args, mode, **kwargs):
    """
    Factory function to get dataset based on arguments.
    
    Args:
        args: Configuration arguments containing dataset and dimension info
        mode: Dataset mode ('train', 'val', 'test')
        **kwargs: Additional keyword arguments, including dataset_name_list
    
    Returns:
        Dataset instance for the specified configuration
    """
    
    if args.dimension == '3d':
        if args.dataset == 'universal':
            from .dim3.dataset_universal import UniversalDataset

            return UniversalDataset(args, dataset_list=kwargs['dataset_name_list'], mode=mode)

        else:
            raise NameError(f"No {args.dataset} dataset")
