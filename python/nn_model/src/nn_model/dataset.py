"""Dataset utilities"""

import bisect
import os
import random

import numpy as np
import numpy.lib.format as npf
import torch
from torch.utils.data import Dataset


class AudioScoreDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Iterates over a list of audio-score pairs"""

    def __init__(
        self,
        file_pairs: list[tuple[str, str]],
        hop_length: int = 256,
        win_length: int = 512,
        seg_length: int = 2048,
        n_classes: int = 4,
    ):
        """Create AudioScoreDataset

        Args:
            file_pairs (list[tuple[str, str]]): List of (audio, score) file pairs:
              audio - path to the npy file with raw audio data
              scores - path to the npy file with raw scores data; The scores should have value from 0 to 1 (or their fractions)
            hop_length (int, optional): Audio data step interval
            win_length (int, optional): The size of fft window
            seg_length (int, optional): The size of the output audio segment
            n_classes (int, optional): The number of scores in the score file

        Raises:
            ValueError: The number of elements in a pair of files differs

        """
        self.file_pairs = file_pairs
        self.hop_length = hop_length
        self.win_length = win_length
        self.seg_length = seg_length
        self.n_classes = n_classes

        files_info = []
        step_offsets = []
        total_steps = 0
        self._arrays: list[tuple[NpyReader, NpyReader]] = []
        for audio, scores in file_pairs:
            audio_reader = NpyReader(audio)
            scores_reader = NpyReader(scores)
            self._arrays.append((audio_reader, scores_reader))
            audio_shape, audio_type = audio_reader.shape, audio_reader.dtype
            scores_shape = scores_reader.shape
            if (
                len(scores_shape) != 2  # noqa: PLR2004
                or audio_shape[0] != scores_shape[0]
                or n_classes != scores_shape[1]
            ):
                msg = f"The scores file '{scores}' contains wrong number of elements, it should have (number of audio samples of '{audio}')*{n_classes} elements"
                raise ValueError(msg)
            n_steps = audio_shape[0] - win_length  # the last window

            n_steps = (
                (n_steps // hop_length)
                if (n_steps % hop_length) == 0
                else (n_steps // hop_length + 1)
            )  # same as `n_steps = int(ceil(n_steps / hop_length))`

            # Multiplier to normalize the audio input
            mult = 1.0
            match audio_type.str:
                case "i2" | "<i2":
                    mult = 1.0 / 0x7FFF
                case "i4" | "<i4":
                    mult = 1.0 / 0x7FFFFFFF

            n_steps += 1  # the last window
            files_info.append((audio_shape[0], n_steps, mult))
            step_offsets.append(total_steps)
            total_steps += n_steps

        self.files_info = files_info
        self.step_offsets = step_offsets
        self.total_steps = total_steps

        self.win_mid = win_length // 2

    def __len__(self):
        """Return number of items in this dataset

        Returns:
            int: Number of items in the dataset

        """
        return self.total_steps

    def _end_element(self, file_idx, step) -> int:
        (size, n_steps, _) = self.files_info[file_idx]
        step_end = step + 1
        return size - (n_steps - step_end) * self.hop_length

    def _make_audio(self, end_el: int, file_idx: int) -> np.ndarray:
        audio = self._arrays[file_idx][0]

        audio_seg = np.zeros(self.seg_length, dtype=np.float32)
        if end_el < self.seg_length:
            copy_offset = self.seg_length - end_el
            audio_seg[copy_offset:] = audio.read(0, end_el)
        else:
            start_el = end_el - self.seg_length
            audio_seg[:] = audio.read(start_el, end_el)

        audio_seg *= self.files_info[file_idx][2]
        return audio_seg

    def _make_scores(self, end_el: int, file_idx: int) -> np.ndarray:
        index = end_el - self.win_mid

        if index < 0:
            return np.zeros(self.n_classes, dtype=np.float32)

        return self._arrays[file_idx][1].read(index, index + 1)[0].astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return audio sample and scores at index

        Args:
            index (int): index of the step

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (audio_segment, target_scores) pair

        """
        file_idx = max(0, bisect.bisect_right(self.step_offsets, index) - 1)
        step = index - self.step_offsets[file_idx]

        end_el = self._end_element(file_idx, step)

        audio = self._make_audio(end_el, file_idx)
        scores = self._make_scores(end_el, file_idx)

        return torch.from_numpy(audio), torch.from_numpy(scores)


class RandomOffsetDataset(AudioScoreDataset):
    """An overload of the AudioScoreDataset that randomly shifts the training data"""

    def _end_element(self, file_idx, step) -> int:
        end_el = super()._end_element(file_idx, step)
        return max(0, end_el - random.randint(0, self.hop_length - 1))


class NpyReader:
    """Reads row ranges of a C-ordered .npy file using plain file I/O."""

    def __init__(self, path: str):
        """Create NpyReader

        Args:
            path (str): Path to the npy file

        Raises:
            ValueError: The file is stored in Fortran order

        """
        self.path = os.fspath(path)
        with open(self.path, "rb") as fp:
            version = npf.read_magic(fp)
            if version[0] == 1:
                shape, fortran_order, dtype = npf.read_array_header_1_0(fp)
            else:
                shape, fortran_order, dtype = npf.read_array_header_2_0(fp)
            self._data_offset = fp.tell()
        if fortran_order and len(shape) > 1:
            msg = f"The file '{path}' is in Fortran order, which is not supported"
            raise ValueError(msg)
        self.shape: tuple[int, ...] = shape
        self.dtype: np.dtype = dtype
        self._row_shape = shape[1:]
        self._row_bytes = dtype.itemsize * int(np.prod(self._row_shape, dtype=np.int64))
        self._fp = None

    def __getstate__(self):
        """Get the state for pickling"""
        state = self.__dict__.copy()
        state["_fp"] = None  # file handles can't be shared with the workers
        return state

    def read(self, start: int, stop: int) -> np.ndarray:
        """Read the rows [start, stop)

        Args:
            start (int): The first row
            stop (int): The row after the last one

        Returns:
            np.ndarray: The rows, shape is (stop - start, *shape[1:])

        """
        out = np.empty((stop - start, *self._row_shape), dtype=self.dtype)
        if self._fp is None:
            self._fp = open(self.path, "rb", buffering=0)  # noqa: SIM115
        self._fp.seek(self._data_offset + start * self._row_bytes)
        view = memoryview(out.reshape(-1).view(np.uint8))
        while len(view) > 0:
            n = self._fp.readinto(view)
            if not n:
                msg = f"Unexpected end of file '{self.path}'"
                raise EOFError(msg)
            view = view[n:]
        return out
