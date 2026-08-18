/* GenRec project page data.
 *
 * Field semantics
 * ---------------
 * sectionA = {
 *   baselineLabel: string  // header for the rightmost column ("GEN3C", "CAT3D", ...)
 *   rows: [{
 *     sceneName:       string   // small caption above the row
 *     input:           string   // path/URL to the input image (leftmost column, fixed)
 *     groundTruth:     string[] // N ground-truth images; slider cycles through them.
 *     ourSamples:      string[] // N output images for our model; slider cycles through them.
 *     baselineSamples: string[] // N baseline images; slider cycles through them.
 *                               // All three arrays should have the same length.
 *                               // length 0 or 1 hides the slider gracefully.
 *     altPrefix:       string   // used to build alt text (e.g. "scene_0042 input image")
 *   }, ...]
 * }
 *
 * sectionB = {
 *   withLabel:    string   // header for the middle column
 *   withoutLabel: string   // header for the right column
 *   rows: [{
 *     sceneName:  string   // small caption above the row
 *     inputs:     string | string[]  // 1..4 input images. 1 -> single tile, 2 -> stack,
 *                                    // 3-4 -> 2x2 grid.
 *     withScm:    string   // path/URL to the "with SCM init" video (.mp4)
 *     withoutScm: string   // path/URL to the "without SCM init" video (.mp4)
 *     poster:     string?  // optional shared poster (still frame) shown until video loads
 *     altPrefix:  string   // used to build alt text
 *   }, ...]
 * }
 *
 * Asset layout suggestion (when you replace the placeholders):
 *   assets/sectionA/<scene>/input.jpg
 *   assets/sectionA/<scene>/gt_0.jpg ... gt_N.jpg
 *   assets/sectionA/<scene>/ours_0.jpg ... ours_N.jpg
 *   assets/sectionA/<scene>/baseline_0.jpg ... baseline_N.jpg
 *   assets/sectionB/<scene>/in_0.jpg ... in_3.jpg
 *   assets/sectionB/<scene>/with_scm.mp4
 *   assets/sectionB/<scene>/without_scm.mp4
 *   assets/sectionB/<scene>/poster.jpg
 *
 * The page also tries a sibling .webp for every .jpg/.png path via <picture>.
 * Add the .webp later and browsers will prefer it; nothing is required up front.
 */

window.sectionA = {
  baselineLabel: "GEN3C",
  rows: [
    {
      sceneName: "RE10K · 0223924f43297881",
      input: "assets/sectionA/re10k_1/0223924f43297881_frame_0000.png",
      groundTruth: [
        "assets/sectionA/re10k_1/gt_0223924f43297881_00.png",
        "assets/sectionA/re10k_1/gt_0223924f43297881_01.png",
        "assets/sectionA/re10k_1/gt_0223924f43297881_02.png",
        "assets/sectionA/re10k_1/gt_0223924f43297881_03.png",
        "assets/sectionA/re10k_1/gt_0223924f43297881_04.png",
        "assets/sectionA/re10k_1/gt_0223924f43297881_05.png",
        "assets/sectionA/re10k_1/gt_0223924f43297881_06.png"
      ],
      ourSamples: [
        "assets/sectionA/re10k_1/ours_0223924f43297881_00.png",
        "assets/sectionA/re10k_1/ours_0223924f43297881_01.png",
        "assets/sectionA/re10k_1/ours_0223924f43297881_02.png",
        "assets/sectionA/re10k_1/ours_0223924f43297881_03.png",
        "assets/sectionA/re10k_1/ours_0223924f43297881_04.png",
        "assets/sectionA/re10k_1/ours_0223924f43297881_05.png",
        "assets/sectionA/re10k_1/ours_0223924f43297881_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/re10k_1/baseline_0223924f43297881_00.png",
        "assets/sectionA/re10k_1/baseline_0223924f43297881_01.png",
        "assets/sectionA/re10k_1/baseline_0223924f43297881_02.png",
        "assets/sectionA/re10k_1/baseline_0223924f43297881_03.png",
        "assets/sectionA/re10k_1/baseline_0223924f43297881_04.png",
        "assets/sectionA/re10k_1/baseline_0223924f43297881_05.png",
        "assets/sectionA/re10k_1/baseline_0223924f43297881_06.png"
      ],
      altPrefix: "scene_0042"
    },
    {
      sceneName: "RE10K · 1aa94a69594df8a6",
      input: "assets/sectionA/re10k_2/1aa94a69594df8a6_frame_0000.png",
      groundTruth: [
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_00.png",
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_01.png",
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_02.png",
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_03.png",
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_04.png",
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_05.png",
        "assets/sectionA/re10k_2/gt_1aa94a69594df8a6_06.png"
      ],
      ourSamples: [
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_00.png",
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_01.png",
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_02.png",
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_03.png",
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_04.png",
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_05.png",
        "assets/sectionA/re10k_2/ours_1aa94a69594df8a6_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_00.png",
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_01.png",
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_02.png",
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_03.png",
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_04.png",
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_05.png",
        "assets/sectionA/re10k_2/baseline_1aa94a69594df8a6_06.png"
      ],
      altPrefix: "scene_0118"
    },
    {
      sceneName: "RE10K · 0de79d4f3d7a9171",
      input: "assets/sectionA/re10k_3/0de79d4f3d7a9171_frame_0000.png",
      groundTruth: [
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_00.png",
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_01.png",
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_02.png",
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_03.png",
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_04.png",
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_05.png",
        "assets/sectionA/re10k_3/gt_0de79d4f3d7a9171_06.png"
      ],
      ourSamples: [
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_00.png",
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_01.png",
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_02.png",
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_03.png",
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_04.png",
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_05.png",
        "assets/sectionA/re10k_3/ours_0de79d4f3d7a9171_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_00.png",
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_01.png",
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_02.png",
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_03.png",
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_04.png",
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_05.png",
        "assets/sectionA/re10k_3/baseline_0de79d4f3d7a9171_06.png"
      ],
      altPrefix: "scene_0233"
    },
    {
      sceneName: "RE10K · 0cda9adbbedd7948",
      input: "assets/sectionA/re10k_4/0cda9adbbedd7948_frame_0000.png",
      groundTruth: [
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_00.png",
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_01.png",
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_02.png",
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_03.png",
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_04.png",
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_05.png",
        "assets/sectionA/re10k_4/gt_0cda9adbbedd7948_06.png"
      ],
      ourSamples: [
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_00.png",
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_01.png",
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_02.png",
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_03.png",
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_04.png",
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_05.png",
        "assets/sectionA/re10k_4/ours_0cda9adbbedd7948_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_00.png",
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_01.png",
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_02.png",
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_03.png",
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_04.png",
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_05.png",
        "assets/sectionA/re10k_4/baseline_0cda9adbbedd7948_06.png"
      ],
      altPrefix: "scene_0348"
    },
    {
      sceneName: "DL3DV-10K · 3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090",
      input: "assets/sectionA/dl3dv_1/3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_frame_0000.png",
      groundTruth: [
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_00.png",
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_01.png",
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_02.png",
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_03.png",
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_04.png",
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_05.png",
        "assets/sectionA/dl3dv_1/gt_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_06.png"
      ],
      ourSamples: [
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_00.png",
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_01.png",
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_02.png",
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_03.png",
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_04.png",
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_05.png",
        "assets/sectionA/dl3dv_1/ours_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_00.png",
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_01.png",
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_02.png",
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_03.png",
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_04.png",
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_05.png",
        "assets/sectionA/dl3dv_1/baseline_3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090_06.png"
      ],
      altPrefix: "dl3dv_3b7529dc"
    },
    {
      sceneName: "DL3DV-10K · 14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49",
      input: "assets/sectionA/dl3dv_2/14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_frame_0000.png",
      groundTruth: [
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_00.png",
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_01.png",
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_02.png",
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_03.png",
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_04.png",
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_05.png",
        "assets/sectionA/dl3dv_2/gt_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_06.png"
      ],
      ourSamples: [
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_00.png",
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_01.png",
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_02.png",
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_03.png",
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_04.png",
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_05.png",
        "assets/sectionA/dl3dv_2/ours_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_00.png",
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_01.png",
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_02.png",
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_03.png",
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_04.png",
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_05.png",
        "assets/sectionA/dl3dv_2/baseline_14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49_06.png"
      ],
      altPrefix: "dl3dv_14eb48a5"
    },
    {
      sceneName: "DL3DV-10K · 093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334",
      input: "assets/sectionA/dl3dv_3/093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_frame_0000.png",
      groundTruth: [
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_00.png",
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_01.png",
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_02.png",
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_03.png",
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_04.png",
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_05.png",
        "assets/sectionA/dl3dv_3/gt_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_06.png"
      ],
      ourSamples: [
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_00.png",
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_01.png",
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_02.png",
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_03.png",
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_04.png",
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_05.png",
        "assets/sectionA/dl3dv_3/ours_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_00.png",
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_01.png",
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_02.png",
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_03.png",
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_04.png",
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_05.png",
        "assets/sectionA/dl3dv_3/baseline_093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334_06.png"
      ],
      altPrefix: "dl3dv_093ef327"
    },
    {
      sceneName: "DL3DV-10K · 41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45",
      input: "assets/sectionA/dl3dv_4/41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_frame_0000.png",
      groundTruth: [
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_00.png",
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_01.png",
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_02.png",
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_03.png",
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_04.png",
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_05.png",
        "assets/sectionA/dl3dv_4/gt_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_06.png"
      ],
      ourSamples: [
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_00.png",
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_01.png",
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_02.png",
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_03.png",
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_04.png",
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_05.png",
        "assets/sectionA/dl3dv_4/ours_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_06.png"
      ],
      baselineSamples: [
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_00.png",
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_01.png",
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_02.png",
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_03.png",
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_04.png",
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_05.png",
        "assets/sectionA/dl3dv_4/baseline_41036716da7efda334c1d434c4141d15642e0e02f881a01b6c8c36f8bea64c45_06.png"
      ],
      altPrefix: "dl3dv_41036716"
    }
  ]
};

window.sectionB = {
  withLabel: "Ours: w/ SCM init",
  withoutLabel: "Ours: w/o SCM init",
  rows: [
    {
      sceneName: "RE10K · 039b153af4fbfba7",
      inputs: ["assets/sectionB/039b153af4fbfba7_frame_0000.png"],
      withScm: "assets/sectionB/039b153af4fbfba7_w_video.mp4",
      withoutScm: "assets/sectionB/039b153af4fbfba7_video.mp4",
      altPrefix: "re10k_039b153a"
    },
    {
      sceneName: "RE10K · 0f12b97e0e4c7e21",
      inputs: ["assets/sectionB/0f12b97e0e4c7e21_frame_0000.png"],
      withScm: "assets/sectionB/0f12b97e0e4c7e21_w_video.mp4",
      withoutScm: "assets/sectionB/0f12b97e0e4c7e21_video.mp4",
      altPrefix: "re10k_0f12b97e"
    }
  ]
};

window.sectionC = {
  rows: [
    {
      sceneName: "RE10K · 0c8b534612a0a776",
      input: "assets/sectionC/0c8b534612a0a776_frame_0000.png",
      video: "assets/sectionC/0c8b534612a0a776_video.mp4",
      altPrefix: "re10k_0c8b5346"
    },
    {
      sceneName: "RE10K · 0bb99505a71035cc",
      input: "assets/sectionC/0bb99505a71035cc_frame_0000.png",
      video: "assets/sectionC/0bb99505a71035cc_video.mp4",
      altPrefix: "re10k_0bb99505"
    },
    {
      sceneName: "RE10K · 0e41af1514f92887",
      input: "assets/sectionC/0e41af1514f92887_frame_0000.png",
      video: "assets/sectionC/0e41af1514f92887_video.mp4",
      altPrefix: "re10k_0e41af15"
    }
  ]
};

window.sectionD = {
  withLabel: "with refinement",
  withoutLabel: "without refinement",
  rows: [
    {
      sourceImage: "assets/sectionD/gt_1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654_00.png",
      fullWith: "assets/sectionD/w_1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654_00.png",
      fullWithout: "assets/sectionD/wo_1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654_00.png",
      cropWith: "assets/sectionD/crop_w_1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654_00.png",
      cropWithout: "assets/sectionD/crop_wo_1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654_00.png",
      caption: "",
      altPrefix: "dl3dv_1de58be5"
    },
    {
      sourceImage: "assets/sectionD/gt_286239bd0d4436b747987c40f8cbfc0675bb69435d4ab19a3f73b9d816d0c09a_00.png",
      fullWith: "assets/sectionD/w_286239bd0d4436b747987c40f8cbfc0675bb69435d4ab19a3f73b9d816d0c09a_00.png",
      fullWithout: "assets/sectionD/wo_286239bd0d4436b747987c40f8cbfc0675bb69435d4ab19a3f73b9d816d0c09a_00.png",
      cropWith: "assets/sectionD/crop_w_286239bd0d4436b747987c40f8cbfc0675bb69435d4ab19a3f73b9d816d0c09a_00.png",
      cropWithout: "assets/sectionD/crop_wo_286239bd0d4436b747987c40f8cbfc0675bb69435d4ab19a3f73b9d816d0c09a_00.png",
      caption: "",
      altPrefix: "dl3dv_286239bd"
    },
    {
      sourceImage: "assets/sectionD/gt_565553aa894be621e8b4773cac288e60ad0c2cf7edb621be62b348c9a0f78380_01.png",
      fullWith: "assets/sectionD/w_565553aa894be621e8b4773cac288e60ad0c2cf7edb621be62b348c9a0f78380_01.png",
      fullWithout: "assets/sectionD/wo_565553aa894be621e8b4773cac288e60ad0c2cf7edb621be62b348c9a0f78380_01.png",
      cropWith: "assets/sectionD/crop_w_565553aa894be621e8b4773cac288e60ad0c2cf7edb621be62b348c9a0f78380_01.png",
      cropWithout: "assets/sectionD/crop_wo_565553aa894be621e8b4773cac288e60ad0c2cf7edb621be62b348c9a0f78380_01.png",
      caption: "",
      altPrefix: "dl3dv_565553aa"
    },
    {
      sourceImage: "assets/sectionD/gt_032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7_00.png",
      fullWith: "assets/sectionD/w_032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7_00.png",
      fullWithout: "assets/sectionD/wo_032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7_00.png",
      cropWith: "assets/sectionD/crop_w_032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7_00.png",
      cropWithout: "assets/sectionD/crop_wo_032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7_00.png",
      caption: "",
      altPrefix: "dl3dv_032dee9f"
    }
  ]
};

/* ---- Quantitative results (transcribed from the NeurIPS submission) ----
 * Values are kept as strings exactly as printed in the paper tables;
 * the renderer parses them numerically to compute the best value per
 * column (columns with dir: null are not ranked). */

(function () {
  var singleViewColumns = [
    { label: "PSNR", dir: "up", group: "Global" },
    { label: "SSIM", dir: "up", group: "Global" },
    { label: "LPIPS", dir: "down", group: "Global" },
    { label: "FID", dir: "down", group: "Global" },
    { label: "PSNR", dir: "up", group: "Observed" },
    { label: "SSIM", dir: "up", group: "Observed" },
    { label: "PSNR", dir: "up", group: "Unobserved" },
    { label: "SSIM", dir: "up", group: "Unobserved" },
    { label: "Inf. Time", dir: null, group: "" }
  ];

  window.resultsData = {
    benchmarks: [
      {
        id: "re10k",
        label: "RealEstate10K",
        columns: singleViewColumns,
        note: "Single-view extrapolation. Global metrics are computed over the full target frame; Observed restricts them to pixels that receive source evidence under the shared observation mask, Unobserved to pixels that do not. Inference time is per scene.",
        rows: [
          { method: "CameraCtrl", values: ["13.40", "0.4424", "0.4761", "57.42", "15.53", "0.4568", "12.40", "0.3935", "~12s"] },
          { method: "ViewCrafter", values: ["13.29", "0.4792", "0.4293", "46.27", "15.93", "0.5196", "11.83", "0.3937", "~200s"] },
          { method: "SEVA", values: ["13.50", "0.4839", "0.4674", "41.64", "15.97", "0.4999", "12.40", "0.4189", "~20s"] },
          { method: "GF", values: ["13.11", "0.3806", "0.4945", "54.58", "15.56", "0.3986", "11.96", "0.3125", "~40s"] },
          { method: "LVSM", values: ["13.81", "0.4094", "0.5186", "57.89", "16.32", "0.4224", "12.61", "0.3435", "~1s"] },
          { method: "GLD", values: ["13.73", "0.4232", "0.4699", "44.29", "16.07", "0.4533", "12.82", "0.3765", "~42s"] },
          { method: "Gen3R", values: ["13.29", "0.4747", "0.5107", "49.46", "15.82", "0.4862", "12.14", "0.4044", "~90s"] },
          { method: "GEN3C", values: ["15.24", "0.5289", "0.3624", "35.68", "18.13", "0.5689", "13.63", "0.4529", "~1200s"] },
          { method: "GenRec", ours: true, values: ["17.10", "0.6071", "0.3220", "33.09", "20.85", "0.6738", "15.01", "0.5053", "~10s"] }
        ]
      },
      {
        id: "dl3dv",
        label: "DL3DV-10K",
        columns: singleViewColumns,
        note: "Single-view extrapolation, same evaluation protocol as RealEstate10K.",
        rows: [
          { method: "CameraCtrl", values: ["13.26", "0.3346", "0.4893", "69.49", "13.45", "0.3418", "12.76", "0.2957", "~12s"] },
          { method: "ViewCrafter", values: ["17.57", "0.5302", "0.3092", "46.36", "18.75", "0.5659", "15.36", "0.4282", "~200s"] },
          { method: "SEVA", values: ["14.58", "0.4300", "0.4359", "43.51", "14.86", "0.4401", "13.92", "0.3819", "~20s"] },
          { method: "GLD", values: ["15.54", "0.4001", "0.3979", "51.85", "16.10", "0.4184", "14.33", "0.3368", "~42s"] },
          { method: "Gen3R", values: ["13.74", "0.3754", "0.5173", "130.10", "14.02", "0.3886", "12.95", "0.3048", "~90s"] },
          { method: "GEN3C", values: ["18.47", "0.5717", "0.2860", "44.54", "19.66", "0.6026", "16.24", "0.4802", "~1200s"] },
          { method: "GenRec", ours: true, values: ["19.80", "0.6195", "0.2188", "32.36", "21.35", "0.6605", "17.25", "0.5087", "~10s"] }
        ]
      },
      {
        id: "mipnerf",
        label: "Mip-NeRF 360",
        badge: "out-of-distribution",
        columns: singleViewColumns,
        note: "Single-view extrapolation on all 9 Mip-NeRF 360 scenes, an out-of-distribution benchmark never seen during training.",
        rows: [
          { method: "CameraCtrl", values: ["10.89", "0.1870", "0.6534", "214.2", "11.31", "0.1869", "10.67", "0.1821", "~12s"] },
          { method: "ViewCrafter", values: ["11.75", "0.2197", "0.5694", "172.5", "14.46", "0.2510", "10.74", "0.1913", "~200s"] },
          { method: "SEVA", values: ["11.52", "0.2231", "0.5857", "139.8", "12.09", "0.2315", "11.30", "0.2171", "~20s"] },
          { method: "GLD", values: ["12.03", "0.1986", "0.5631", "162.1", "13.32", "0.2044", "11.45", "0.1900", "~42s"] },
          { method: "Gen3R", values: ["11.07", "0.2056", "0.6564", "223.6", "11.21", "0.2072", "10.96", "0.2015", "~90s"] },
          { method: "GEN3C", values: ["12.86", "0.2343", "0.5796", "207.4", "14.43", "0.2608", "12.12", "0.2117", "~1200s"] },
          { method: "GenRec", ours: true, values: ["13.10", "0.2478", "0.4895", "139.4", "15.37", "0.2858", "12.16", "0.2200", "~10s"] }
        ]
      },
      {
        id: "interp",
        label: "Two-View Interpolation",
        columns: [
          { label: "PSNR", dir: "up", group: "RealEstate10K" },
          { label: "SSIM", dir: "up", group: "RealEstate10K" },
          { label: "LPIPS", dir: "down", group: "RealEstate10K" },
          { label: "PSNR", dir: "up", group: "DL3DV-10K" },
          { label: "SSIM", dir: "up", group: "DL3DV-10K" },
          { label: "LPIPS", dir: "down", group: "DL3DV-10K" }
        ],
        note: "Two-view interpolation: the first and last frames of each trajectory are given as sources, so the task is nearly reconstruction-only.",
        rows: [
          { method: "CameraCtrl", values: ["12.87", "0.4271", "0.5022", "15.60", "0.3456", "0.4705"] },
          { method: "ViewCrafter", values: ["12.61", "0.4550", "0.4573", "17.38", "0.5214", "0.3087"] },
          { method: "SEVA", values: ["16.15", "0.5833", "0.3477", "21.38", "0.6906", "0.1845"] },
          { method: "LVSM", values: ["16.88", "0.5493", "0.3983", "21.31", "0.6801", "0.2223"] },
          { method: "GLD", values: ["17.64", "0.5707", "0.2900", "19.49", "0.5575", "0.2376"] },
          { method: "Gen3R", values: ["12.17", "0.4385", "0.5697", "13.99", "0.3813", "0.5030"] },
          { method: "GEN3C", values: ["15.70", "0.5632", "0.3282", "21.16", "0.6869", "0.2137"] },
          { method: "GenRec", ours: true, values: ["19.80", "0.7043", "0.2205", "22.02", "0.7009", "0.1558"] }
        ]
      }
    ]
  };

  window.componentAblation = {
    columns: [
      { label: "PSNR", dir: "up", group: "Global" },
      { label: "SSIM", dir: "up", group: "Global" },
      { label: "LPIPS", dir: "down", group: "Global" },
      { label: "FID", dir: "down", group: "Global" },
      { label: "PSNR", dir: "up", group: "Observed" },
      { label: "SSIM", dir: "up", group: "Observed" }
    ],
    datasets: [
      {
        label: "RealEstate10K",
        delta: "The VAE skips and LoRA primarily improve generative quality (FID); the reconstruction branch then adds +1.62 dB observed PSNR without sacrificing it.",
        rows: [
          { method: "Generative Backbone", values: ["15.86", "0.5483", "0.3402", "37.82", "19.20", "0.6139"] },
          { method: "+ VAE Skip & LoRA", values: ["15.86", "0.5591", "0.3394", "35.27", "19.23", "0.6152"] },
          { method: "+ Reconstruction Branch", ours: true, values: ["17.10", "0.6071", "0.3220", "33.09", "20.85", "0.6738"] }
        ]
      },
      {
        label: "DL3DV-10K",
        delta: "On DL3DV the reconstruction branch is worth +1.33 dB observed PSNR while further improving FID by 4.5 points.",
        rows: [
          { method: "Generative Backbone", values: ["18.62", "0.5796", "0.2556", "39.94", "19.96", "0.6163"] },
          { method: "+ VAE Skip & LoRA", values: ["18.64", "0.5828", "0.2512", "36.90", "20.02", "0.6215"] },
          { method: "+ Reconstruction Branch", ours: true, values: ["19.80", "0.6195", "0.2188", "32.36", "21.35", "0.6605"] }
        ]
      }
    ]
  };
})();
