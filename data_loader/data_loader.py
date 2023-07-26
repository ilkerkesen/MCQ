from base import MultiDistBaseDataLoaderExplicitSplit, BaseDataLoaderExplicitSplit
from data_loader.transforms import init_transform_dict
from data_loader.ConceptualCaptions_dataset import ConceptualCaptions3M
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.WebVid_dataset import WebVid
from data_loader.VLBench_dataset import VLBench, _collate_fn


def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   question,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='cv2',
                   metadata_filename=None,
                   quva_dir=None,
                   something_something_dir=None,
                   youtube_dir=None,
                   proficiency=False,
                   ):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        question=question,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "WebVid":
        dataset = WebVid(**kwargs)
    elif dataset_name == "ConceptualCaptions3M":
        dataset = ConceptualCaptions3M(**kwargs)
    elif dataset_name == "VLBench":
        dataset = VLBench(
            **kwargs,
            metadata_filename=metadata_filename,
            quva_dir=quva_dir,
            something_something_dir=something_something_dir,
            youtube_dir=youtube_dir,
            proficiency=proficiency,
        )
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class MultiDistTextVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):

    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 question,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True,
                 metadata_filename=None,
                 quva_dir=None,
                 something_something_dir=None,
                 youtube_dir=None,
                 proficiency=False,
                 ):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(
            dataset_name,
            text_params,
            video_params,
            data_dir,
            question,
            metadata_dir=metadata_dir,
            split=split,
            tsfms=tsfm,
            cut=cut,
            subsample=subsample,
            sliding_window_stride=sliding_window_stride,
            reader=reader,
            metadata_filename=metadata_filename,
            quva_dir=quva_dir,
            something_something_dir=something_something_dir,
            youtube_dir=youtube_dir,
            proficiency=proficiency,
        )

        collate_fn = None
        if type(dataset) is VLBench:
            collate_fn = _collate_fn

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.dataset_name = dataset_name


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 question,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True,
                 metadata_filename=None,
                 quva_dir=None,
                 something_something_dir=None,
                 youtube_dir=None,
                 proficiency=False,
                 ):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(
            dataset_name,
            text_params,
            video_params,
            data_dir,
            question,
            metadata_dir=metadata_dir,
            split=split,
            tsfms=tsfm,
            cut=cut,
            subsample=subsample,
            sliding_window_stride=sliding_window_stride,
            reader=reader,
            metadata_filename=metadata_filename,
            quva_dir=quva_dir,
            something_something_dir=something_something_dir,
            youtube_dir=youtube_dir,
            proficiency=proficiency,
        )

        collate_fn = None
        if type(dataset) is VLBench:
            collate_fn = _collate_fn

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.dataset_name = dataset_name


class TextVideoDataLoader_CLIP(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 question,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True,
                 metadata_filename=None,
                 quva_dir=None,
                 something_something_dir=None,
                 youtube_dir=None,
                 proficiency=False,
                 ):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict_clip(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(
            dataset_name,
            text_params,
            video_params,
            data_dir,
            question,
            metadata_dir=metadata_dir,
            split=split,
            tsfms=tsfm,
            cut=cut,
            subsample=subsample,
            sliding_window_stride=sliding_window_stride,
            reader=reader,
            metadata_filename=metadata_filename,
            quva_dir=quva_dir,
            something_something_dir=something_something_dir,
            youtube_dir=youtube_dir,
            proficiency=proficiency,
        )

        collate_fn = None
        if type(dataset) is VLBench:
            collate_fn = _collate_fn

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.dataset_name = dataset_name
