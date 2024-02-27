import os
from typing import Any

import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
from pandas import DataFrame, Series, concat, read_pickle
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import BatchEncoding, XLMRobertaTokenizer


class YorubaNewsclassDataset(Dataset):
    def __init__(self, df: DataFrame) -> None:
        """
        Initalizer for the IMDBDataset class. This class takes as input a
        pandas dataframe and handles the logic for creating a torch-like
        dataset.

        ---parameters---
        df: pandas.DataFrame containing the preprocessed data.
        """
        self.df = df
        self.df["attention_mask"] = self.df["attention_mask"].apply(lambda x: x.astype("int64"))
        self.df["token_type_ids"] = self.df["token_type_ids"].apply(lambda x: x.astype("int64"))
        self.df["input_ids"] = self.df["input_ids"].apply(lambda x: x.astype("int64"))

    def __len__(self) -> int:
        """
        Returns the length of the dataset as the length of the underlying
        pandas dataframe.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Handles getting data from the dataset through indexing by
        returning a dictionary of the underlying pandas dataframe
        at the specfied index.

        ---parameters---
        idx: interger value representing the row index of the dataset

        ---output---
        A dictionary containing the label, input_ids, token_type_ids,
        attention_mask and label_mask of the pandas dataframe at the
        specified index. These are the columns necessary for running
        the model.
        """
        return {
            "labels": self.df["label"][idx],
            "input_ids": self.df["input_ids"][idx],
            "token_type_ids": self.df["token_type_ids"][idx],
            "attention_mask": self.df["attention_mask"][idx],
            "label_mask": self.df["label_mask"][idx],
        }


class YorubaNewsclassDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_labeled_examples: float = 1.0,
        upsample_labeled: bool = False,
        downsample_unlabeled: bool = True,
        max_seq_length: int = 64,
    ) -> None:
        """
        Initalilzer contains the logic for
        downloading to disk, preprocessing, loading from disk and staging
        the imdb dataset.

        ---parameters---
        batch_size: integer value representing the size to batch the original data

        annotation_frac: float value representing what fraction of the total
        IMDB data should be labeled.

        upsample_labeled: Boolean value specifying if upsampling should be
        applied to the labeled examples. Aims to prevent having batches of purely
        unlabeled examples as specified in https://aclanthology.org/2020.acl-main.191

        downsample_unlabeled: Boolean value specifying if downsampling should be
        applied to the unlabeled examples. Aims to prevent having batches of purely
        unlabeled examples as specified in https://aclanthology.org/2020.acl-main.191

        max_seq_length: Integer value between [1, 512] specifying the maximum sequence
        length to apply when tokenizing the text data.
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_labeled_examples = num_labeled_examples
        self.upsample_labeled = upsample_labeled
        self.downsample_unlabeled = downsample_unlabeled
        if self.upsample_labeled is True and self.downsample_unlabeled is True:
            raise Exception("Cannot upsample labeled and also downsample unlabeled")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            "castorini/afriberta_large"
        )  # automodel
        self.tokenizer.model_max_length = 512
        self.label_map = {
            "nigeria": 0,
            "politics": 1,
            "entertainment": 2,
            "world": 3,
            "health": 4,
            "africa": 5,
            "sport": 6,
        }

    def preprocess_function(self, examples: Series) -> BatchEncoding:
        """
        Applies the hugging faces tokenizer's encode plus function on a given
        text input.

        ---parameters---
        examples: A row from a dataset or dictionary like object containing the
        column "text" with text input.
        """
        return self.tokenizer.encode_plus(
            text=examples["news_title"],
            add_special_tokens=True,  # Adds [CLS] and [SEP] tokens
            truncation=True,
            # Truncates to maximum acceptable input length (or max_len argument if provided)
            padding="max_length",  # pads to max_length argument
            max_length=self.max_seq_length,
            # return_tensor='pt', #return a pytorch array, not compatible with arrowdataset
            return_token_type_ids=True,
            return_attention_mask=True,
        )

    def download_data(
        self,
        dataset: str,
        path: str,
        split: str | None,
        output: bool = True,
        data_files: str | list[str] | None = None,
    ) -> None | DataFrame:
        """
        Function for downloading onto disk in pickle format the raw data
        from the huggingfaces repository.

        ---parameters---
        dataset: string value specifying the path on the huggingfaces repository
        to the dataset to be downloaded.

        path: string value specifying where on disk to save data.

        split: string value specifying which split of the dataset to
        download.

        output: boolean value indicating if the function should return the
        downloaded data.

        ---output---
        pandas.Dataframe containing the downloaded IMDB data if output parameter
        set to True.
        """
        # split may be either train test unsupervised
        df = load_dataset(dataset, split=split, data_files=data_files)
        df = df["train"]
        df = df.map(self.preprocess_function)  # apply the preprocess function
        df = df.to_pandas()  # convert the arrow dataset to a pandas dataframe
        if split == "unsupervised":
            df["label"] = -1
        else:
            df["label"] = df["label"].apply(lambda x: self.label_map[x])
        df.to_pickle(path)  # pickle the pandas dataframe on disk

        # according to the split assign the dataframe the corresponding variable
        if split == "train":
            self.train = df
        if split == "test":
            self.test = df
        if split == "unsupervised":
            self.unsupervised = df

        if output:
            return df

        return None

    def load_pickle_data(
        self, split: str, path: str | None = None, df: DataFrame | None = None
    ) -> None:
        """
        Function handling loading a pickled pandas dataframe in pickle from
        disk or alternatively a pandas dataframe variable and staging it.

        ---parameters---
        split: string value specifying which split of the dataset is loaded.

        path: string value specifying where on disk to load the pickled pandas
        dataframe.
        """
        # from disk or variable
        if path is not None and df is None:
            try:
                df = read_pickle(path)
            except:
                print("Data not found at path...")
        elif path is None and df is not None:
            df = df
        else:
            df = None

        # how to stage the data
        if split == "train":
            if df is None:
                response = input("Download data from online repo? [y/n] ")
                if response != "y":
                    raise Exception("No data found and download rejected...")
                print("Downloading data...")
                df = self.download_data(
                    dataset="mtek2000/yoruba_newsclass_topic",
                    path=path if path is not None else os.getcwd(),
                    split=None,
                    output=True,
                    data_files=["train_clean.csv"],
                )
            self.train = df
        if split == "few_shot":
            if df is None:
                response = input("Download data from online repo? [y/n] ")
                if response != "y":
                    raise Exception("No data found and download rejected...")
                print("Downloading data...")
                df = self.download_data(
                    dataset="mtek2000/yoruba_newsclass_topic",
                    path=path if path is not None else os.getcwd(),
                    split=None,
                    output=True,
                    data_files=["train_clean.csv"],
                )
            self.few_shot = df
        if split == "test":
            if df is None:
                response = input("Download data from online repo? [y/n] ")
                if response != "y":
                    raise Exception("No data found and download rejected...")
                print("Downloading data...")
                df = self.download_data(
                    dataset="mtek2000/yoruba_newsclass_topic",
                    path=path if path is not None else os.getcwd(),
                    split=None,
                    output=True,
                    data_files=["test.csv", "dev.csv"],
                )
            self.test = df
        if split == "unsupervised":
            if df is None:
                response = input("Download data from online repo? [y/n] ")
                if response != "y":
                    raise Exception("No data found and download rejected...")
                print("Downloading data...")
                df = self.download_data(
                    dataset="mtek2000/yoruba_newsclass_topic",
                    path=path if path is not None else os.getcwd(),
                    split=None,
                    output=True,
                    data_files="train_noisy.csv",
                )
            self.unsupervised = df

    def prepare_data(self, split: str) -> None:  # type: ignore
        """
        Function handling how to prepare the preprocessed data already staged
        in the class to be used for training or testing/validation.
        """

        if not hasattr(self, split):
            raise Exception("Please load or download the data for this split first")

        # logic for handling train data
        if split == "train":
            # if we are using less than all of the data take a random subsample
            # if not all of the train data should be labeled...
            if self.num_labeled_examples <= len(self.train) and (
                self.upsample_labeled is True or self.downsample_unlabeled is True
            ):
                # Split the dataset into two according to the annotation frac size
                annotated = self.train.sample(n=self.num_labeled_examples).reset_index(drop=True)
                unannotated = self.train.drop(annotated.index)

                # apply a label mask to each split, this will be used to tell the
                # model if a labelled loss should be calculated for this example
                # when training downstream
                annotated["label_mask"] = 1
                unannotated["label_mask"] = 0
                unannotated["label"] = -1

                # If we want to upsample the labeled examples following the split
                if self.upsample_labeled is True:
                    # Calculate how many times to replicate each labeled example
                    balance = max(
                        1, int(np.log2(1 / (self.num_labeled_examples / len(self.train))))
                    )
                    # concatenate the annotated data balance times together
                    # with the unannotated data and shuffle
                    self.train = concat([annotated for i in range(balance)] + [unannotated])

                # If we want to downsample the labeled examples following the split and we have unlabeled data
                elif self.downsample_unlabeled is True and len(self.unsupervised) > 0:
                    self.unsupervised["label"] = -1
                    # join all the unlabled data
                    unannotated = concat([unannotated, self.unsupervised])

                    # set the label and label mask for the model
                    unannotated["label"] = -1
                    unannotated["label_mask"] = 0

                    # sample from the unlabeled dataset
                    if len(unannotated) < len(annotated):
                        unannotated = unannotated.sample(frac=1).reset_index(drop=True)
                    else:
                        unannotated = unannotated.sample(n=len(annotated)).reset_index(drop=True)

                    # save to train such that for labeled and unlabeled examples are ordered one after the other
                    self.train = (
                        concat([annotated, unannotated])
                        .sort_index(kind="mergesort")
                        .reset_index(drop=True)
                    )
                else:
                    # this sample job is run to equate the number of RNG in all branches of the case condition
                    unannotated.sample(n=len(annotated)).reset_index(drop=True)
                    self.train = annotated
            else:
                self.train = self.train.sample(n=self.num_labeled_examples).reset_index(drop=True)
                self.train["label_mask"] = 1

        if split == "test":
            self.test["label_mask"] = 1

        if split == "unsupervised":
            # dummy label will not be used for loss calculations
            self.unsupervised["label"] = -1
            self.unsupervised["label_mask"] = 0

        # assign the same number of examples per class
        if split == "few_shot":
            labeled = []
            for i in range(len(self.label_map)):
                labeled.append(
                    self.few_shot[self.few_shot["label"] == i].sample(n=self.num_labeled_examples)
                )

            # assign labels
            annotated = concat(labeled).reset_index(drop=True)
            annotated["label_mask"] = 1

            # drop remainder
            remainder = self.few_shot.drop(annotated.index)

            # if we are using unlabeled data...
            if self.upsample_labeled is True or self.downsample_unlabeled is True:
                if not hasattr(self, "unsupervised"):
                    raise Exception(
                        "Trying to upsample or downsample but no unsupervised dataset found..."
                    )
                # full unlabeled dataset
                unannotated = concat([remainder, self.unsupervised])

                # generate a batch the same size as the total few shot batch
                unannotated = unannotated.sample(
                    n=int(len(self.label_map) * self.num_labeled_examples)
                ).reset_index(drop=True)

                # assign dummy label and mask
                unannotated["label_mask"] = 0
                unannotated["label"] = -1

                # concat the labaled and unlabeled data to be interwoven
                self.few_shot = (
                    concat([annotated, unannotated])
                    .sort_index(kind="mergesort")
                    .reset_index(drop=True)
                )

            else:
                # equate number of seed operations in both paths
                remainder.sample(frac=1)
                self.few_shot = annotated
        return

    def setup(self, stage: str) -> None:
        """
        Function that transforms the staged data into torch dataset objects
        according to the training stage

        ---parameters---
        stage: string value, either "fit", "test", "predict", indicating what
        stage of training the data should be prepared for.
        """
        if stage == "fit":
            self.train = YorubaNewsclassDataset(self.train)
            self.val = YorubaNewsclassDataset(self.test)

        if stage == "few_shot":
            self.train = YorubaNewsclassDataset(self.few_shot)
            self.val = YorubaNewsclassDataset(self.test)

        if stage == "test":
            self.test = YorubaNewsclassDataset(self.test)

        if stage == "predict":
            self.predict = YorubaNewsclassDataset(self.test)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the torch data loader for the training set with
        random sampling applied.
        """
        sampler = RandomSampler(self.train)
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=11,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the torch data loader for the validation set with
        random sampling applied.
        """
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=11, persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the torch data loader for the test set with
        random sampling applied.
        """
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=11, persistent_workers=True
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the torch data loader for the prediction set with
        random sampling applied.
        """
        return DataLoader(
            self.predict, batch_size=self.batch_size, num_workers=11, persistent_workers=True
        )
