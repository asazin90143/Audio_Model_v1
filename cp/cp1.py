def process_audio(input_wav_path, array_geometry=None, sample_rate=24000):
    audio, sr, C = load_audio(input_wav_path)           # shape: [T, C]
    audio = resample(audio, sr, sample_rate)
    audio = rms_normalize(audio)

    denoised = dae_denoise(audio)                       # preserves phase; shape [T, C]

    sel_results = None
    if C > 1:
        feats = build_seld_features(denoised, array_geometry)  # GCC-PHAT, IPD, log-mels
        sel_results = seld_model_infer(feats)                  # events + DOA
        sel_tracks = track_association(sel_results)

    sed_events = sed_model_infer(extract_sed_features(denoised))  # (class, t_start, t_end, score)
    event_tracks = align_events_with_sel(sed_events, sel_tracks) if sel_results else sed_events

    separated_tracks = []
    if C > 1:
        for track in event_tracks:
            bf = init_beamformer(array_geometry, track.doa)     # MVDR/GEV steering
            sep = spatial_mask_net(denoised, bf, track)         # DOA-conditioned masks
            separated_tracks.append((track, sep))
    else:
        sep_list = mono_bss_separate(denoised[:,0], sed_events) # SepFormer/Conv-TasNet with class conditioning
        separated_tracks.extend(sep_list)                        # [(event, waveform), ...]

    enhanced_outputs = []
    for event, wav in separated_tracks:
        enh = class_specific_enhance(wav, event.class_label)    # SEGAN, transient model, etc.
        final = loudness_normalize(true_peak_limit(enh))
        out_path = write_wav(final, make_filename(event))
        enhanced_outputs.append((event, out_path))

    return sed_events, enhanced_outputs