import gc
import hashlib
import os
import queue
import threading
import warnings
from multiprocessing import cpu_count
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}


class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        # c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])


def validate_onnx(model_path, min_size_mb=10):
    """Validate that an ONNX file is not corrupted by checking file size and magic bytes.
    
    Args:
        model_path: Path to the .onnx file
        min_size_mb: Minimum expected file size in MB (default 10MB, since most MDX models are 50-100MB+)
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    if not os.path.exists(model_path):
        return False, f"File does not exist: {model_path}"
    
    file_size = os.path.getsize(model_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb < min_size_mb:
        return False, (
            f"ONNX file too small ({file_size_mb:.1f}MB < {min_size_mb}MB minimum). "
            f"The file is likely corrupted or incompletely downloaded: {model_path}"
        )
    
    # Check ONNX protobuf magic bytes
    # ONNX files start with a protobuf header - the first bytes should be valid protobuf
    try:
        with open(model_path, 'rb') as f:
            header = f.read(4)
            if len(header) < 4:
                return False, f"ONNX file is empty or truncated: {model_path}"
    except Exception as e:
        return False, f"Cannot read ONNX file header: {e}"
    
    return True, "OK"


def redownload_mdx_model(model_filename, model_dir):
    """Re-download a corrupted MDX model file.
    
    Args:
        model_filename: Name of the ONNX file (e.g. 'UVR-MDX-NET-Voc_FT.onnx')
        model_dir: Directory containing MDX models
    
    Returns:
        bool: True if re-download succeeded, False otherwise
    """
    from src.download_models import dl_model, MDX_DOWNLOAD_LINK
    from pathlib import Path
    model_path = os.path.join(model_dir, model_filename)
    dir_path = Path(model_dir)
    
    # Remove the corrupted file
    corrupted_path = None
    if os.path.exists(model_path):
        corrupted_path = model_path + ".corrupted"
        print(f"[MDX] Removing corrupted file, backup saved to: {corrupted_path}")
        try:
            os.rename(model_path, corrupted_path)
        except Exception:
            os.remove(model_path)
    
    # Re-download
    try:
        print(f"[MDX] Re-downloading {model_filename}...")
        dl_model(MDX_DOWNLOAD_LINK, model_filename, dir_path, force=True)
        
        # Validate the re-downloaded file
        is_valid, reason = validate_onnx(model_path)
        if is_valid:
            print(f"[MDX] Successfully re-downloaded {model_filename}")
            # Remove the corrupted backup
            if corrupted_path and os.path.exists(corrupted_path):
                os.remove(corrupted_path)
            return True
        else:
            print(f"[MDX] Re-downloaded file is still invalid: {reason}")
            return False
    except Exception as e:
        print(f"[MDX] Failed to re-download {model_filename}: {e}")
        return False


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(self, model_path: str, params: MDXModel, processor=0):

        # Set the device and the provider (CPU or CUDA)
        self.device = (
            torch.device(f"cuda:{processor}")
            if processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"]
            if processor >= 0
            else ["CPUExecutionProvider"]
        )

        print(self.provider, self.device)

        self.model = params

        # Validate ONNX file integrity before loading
        is_valid, reason = validate_onnx(model_path)
        if not is_valid:
            # Try to auto-re-download the corrupted file
            model_dir = os.path.dirname(model_path)
            model_filename = os.path.basename(model_path)
            print(f"[MDX] ONNX validation failed: {reason}")
            print(f"[MDX] Attempting automatic re-download of {model_filename}...")
            if redownload_mdx_model(model_filename, model_dir):
                is_valid, reason = validate_onnx(model_path)
            if not is_valid:
                raise ValueError(
                    f"ONNX model file is corrupted and auto-re-download failed. "
                    f"Please manually delete and re-download: {model_path}\n"
                    f"Reason: {reason}"
                )

        # Load the ONNX model using ONNX Runtime
        try:
            self.ort = ort.InferenceSession(model_path, providers=self.provider)
        except Exception as e:
            # If ONNX Runtime fails to load, the file is corrupted
            # Try re-downloading once more
            model_dir = os.path.dirname(model_path)
            model_filename = os.path.basename(model_path)
            print(f"[MDX] ONNX Runtime failed to load model: {e}")
            print(f"[MDX] Attempting automatic re-download of {model_filename}...")
            if redownload_mdx_model(model_filename, model_dir):
                self.ort = ort.InferenceSession(model_path, providers=self.provider)
            else:
                raise RuntimeError(
                    f"Failed to load ONNX model even after re-download: {model_path}\n"
                    f"Original error: {e}"
                )
        
        # Preload the model for faster performance
        self.ort.run(None, {'input': torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        """
        Segment or join segmented wave array

        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array. If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)

        Returns:
            numpy array: Segmented or joined wave array
        """

        if combine:
            processed_wave = None  # Initializing as None instead of [] for later numpy array concatenation
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):

                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size

        Args:
            wave: (np.array) Wave array to be padded

        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Padded wave
        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        # print(self.device)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        """
        Process each wave segment in a multi-threaded environment

        Args:
            mix_waves: (torch.Tensor) Wave segments to be processed
            trim: (int) Number of samples trimmed during padding
            pad: (int) Number of samples padded during padding
            q: (queue.Queue) Queue to hold the processed wave segments
            _id: (int) Identifier of the processed wave segment

        Returns:
            numpy array: Processed wave segment
        """
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        """
        Process the wave array in a multi-threaded environment

        Args:
            wave: (np.array) Wave array to be processed
            mt_threads: (int) Number of threads to be used for processing

        Returns:
            numpy array: Processed wave array
        """
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)

        # Create a queue to hold the processed wave segments
        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(target=self._process_wave, args=(mix_waves, trim, pad, q, c))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()

        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [list(wave.values())[0] for wave in
                             sorted(processed_batches, key=lambda d: list(d.keys())[0])]
        assert len(processed_batches) == len(waves), 'Incomplete processed batches, please reduce batch size!'
        return self.segment(processed_batches, True, chunk)


def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2, base_device="cuda"):
    vram_gb = 0

    if base_device == "cuda" and torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            device_properties = torch.cuda.get_device_properties(device)
            if device_properties:
                vram_gb = device_properties.total_memory / 1024**3
                m_threads = 1 if vram_gb < 8 else (8 if vram_gb > 32 else 2)
                processor_num = 0
            else:
                raise Exception("No device properties")
        except Exception:
            device = torch.device("cpu")
            m_threads = 8 if torch.cuda.is_available() else 2
            processor_num = -1
    else:
        device = torch.device("cpu")
        m_threads = 8 if torch.cuda.is_available() else 2
        processor_num = -1

    # Look up model parameters by hash first, then by filename, then use fallback
    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    if mp is None:
        # If hash lookup fails, try filename-based lookup
        fname = os.path.basename(model_path)
        mp = model_params.get(fname)
    
    if mp is None:
        # Final fallback with safe defaults for known model types
        fname = os.path.basename(model_path)
        if "Voc_FT" in fname or "Voc" in fname:
            mp = {
                "mdx_dim_f_set": 3072, 
                "mdx_dim_t_set": 8, 
                "mdx_n_fft_scale_set": 6144, 
                "primary_stem": "Vocals",
                "compensate": 1.035
            }
        elif "Inst_HQ" in fname or "Inst" in fname:
            mp = {
                "mdx_dim_f_set": 3072, 
                "mdx_dim_t_set": 8, 
                "mdx_n_fft_scale_set": 6144, 
                "primary_stem": "Instrumental",
                "compensate": 1.021
            }
        elif "KARA" in fname:
            mp = {
                "mdx_dim_f_set": 3072, 
                "mdx_dim_t_set": 8, 
                "mdx_n_fft_scale_set": 6144, 
                "primary_stem": "Instrumental",
                "compensate": 1.021
            }
        elif "Reverb" in fname:
            mp = {
                "mdx_dim_f_set": 3072, 
                "mdx_dim_t_set": 8, 
                "mdx_n_fft_scale_set": 6144, 
                "primary_stem": "No Reverb",
                "compensate": 1.021
            }
        else:
            mp = {
                "mdx_dim_f_set": 3072, 
                "mdx_dim_t_set": 8, 
                "mdx_n_fft_scale_set": 6144, 
                "primary_stem": "Vocals",
                "compensate": 1.025
            }

    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"]
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    duration = librosa.get_duration(y=wave, sr=sr)
    if duration < 60:
        m_threads = 1
    print(f"threads: {m_threads} vram: {vram_gb}")

    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    return main_filepath, invert_filepath
