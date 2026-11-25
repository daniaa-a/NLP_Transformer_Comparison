# Duplicate Review Leakage Analysis

**Scope:** read-only analysis of the current aclImdb files and the existing deterministic split. No dataset, loader, Model A artifact, checkpoint, or metric was modified. Full DistilBERT training was not run.

## Method

The train/validation file assignment was reconstructed from `data_loader.load_reviews("train")` ordering (sorted `neg` files followed by sorted `pos` files) and the loader's `train_test_split(test_size=0.20, random_state=42, stratify=labels, shuffle=True)`. It was cross-checked against the returned loader rows and labels. Raw-byte SHA-256 was computed from each `.txt` file. Loader-text equality uses UTF-8 decoding with replacement and Python universal-newline handling, matching `Path.read_text` in the shared loader. Normalized equality is Unicode NFKC + casefold + collapsed whitespace. Filenames below are relative to `data/raw/aclImdb/`; labels are 0=Negative, 1=Positive.

## Findings

- Labeled files analyzed: 50,000 (25,000 original train and 25,000 official test).
- Unique normalized labeled review texts after deduplication across all 50,000 files: **49,582**; repeated files removed by global deduplication: **418**.
- Unique normalized texts that cross at least one current split boundary: **149 groups**. Exact loader-text cross-boundary groups: **149**. Normalization-only cross-boundary groups beyond those exact matches: **0 pair-group matches**.
- Cross-boundary groups appearing in more than two splits: **0**.
- Cross-boundary groups with conflicting sentiment labels: **0**. Conflicting-label groups anywhere in the full corpus after normalization: **0**.
- The existing loader includes the repeated files in its returned frames; it performs no content-level deduplication. The loader output was independently matched against the reconstructed source-file assignment.

### Pairwise overlaps

| Split pair | Exact loader-text groups | Files in first split | Files in second split | Normalization-only groups |
|---|---:|---:|---:|---:|
| train / validation | 26 | 26 | 28 | 0 |
| train / test | 103 | 103 | 103 | 0 |
| validation / test | 20 | 20 | 20 | 0 |

The measured exact group counts reproduce the earlier report: 26 train/validation, 103 train/test, and 20 validation/test. All exact cross-split groups were byte-for-byte identical in this scan; none required normalization to match. No normalized-only cross-split group was found.

### Affected split files

| Split | Affected files | Split size | Percent affected |
|---|---:|---:|---:|
| train | 129 | 20,000 | 0.645% |
| validation | 48 | 5,000 | 0.960% |
| test | 123 | 25,000 | 0.492% |

There are **300 affected file instances** across the three disjoint returned splits. This is different from the **149 unique cross-split review-text groups**. Because no group spans all three splits, the pairwise group counts add to the total group count.

### Label balance after global normalized deduplication

- Unique non-conflicting negative review texts: 24,698.
- Unique non-conflicting positive review texts: 24,884.
- Conflicting-label normalized text groups: 0.

## Per-review overlap inventory

Each row is one unique exact review-text group crossing a split boundary. The file columns list every matching source file in that split, including multiple copies within one side of a pair. The SHA-256 is over the raw file bytes; for byte-identical rows, the value is shared by all listed files.

| Pair | Raw SHA-256 | Train files (label) | Validation files (label) | Test files (label) | Match type |
|---|---|---|---|---|---|
| train / validation | `114f7b845fe082c72ad71fa5af5ae3d1e6994eccdbb671e4a568882a17b2475c` | `train/neg/744_1.txt` (label=0) | `train/neg/743_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `1bae9d60ed788d77a631030c4dd0c18294046c7e1de49190c4a59c47d8645202` | `train/neg/5551_1.txt` (label=0) | `train/neg/11975_1.txt` (label=0)<br>`train/neg/5953_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `1cc49509cbc2f2bfd442f362ab5426c35365538581d98b4a3ed3c63028a279b8` | `train/pos/1309_10.txt` (label=1) | `train/pos/1307_10.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `233d19f6b8c4d1dc162d1df3c1a9401dd48ff6963bd2d6ab83c2bc822c2bda5c` | `train/neg/6639_1.txt` (label=0) | `train/neg/5082_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `2a0cffb721f91af792c83a5087c2b54d241672599317a22504aae35b96a132dc` | `train/pos/4254_9.txt` (label=1) | `train/pos/4255_9.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `30af065fef4238ea93a349f86708e729ba1615a44a15ac0535e67722efd91cb0` | `train/neg/6827_4.txt` (label=0) | `train/neg/7645_4.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `3e8b1f63f429c4cec3b221647c01250cae7dcbe2aa89c25513e71c05474b61aa` | `train/neg/5083_1.txt` (label=0) | `train/neg/6640_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `4301f3f41c01bb191d50406cc199b7f77d3f8ebc90ace3eaba0df84ffafa0232` | `train/pos/6367_7.txt` (label=1) | `train/pos/6368_7.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `45d0a84d93fa02325e33896296df98bde117f2afffb490337ae3902c37543017` | `train/neg/12447_4.txt` (label=0) | `train/neg/12449_3.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `4c21d4a83077d68839946b722eac556d160267b69aef2a0ebf4e4f42c4252c9c` | `train/neg/985_4.txt` (label=0) | `train/neg/4102_4.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `4df3f37fc12f4bce3119d2a1deec2bca5582e29021597f65bf8caac78e0d648a` | `train/pos/8657_9.txt` (label=1) | `train/pos/8654_9.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `85ea53e459ac88be24eb75065422911a952cd7bf7c6ba74de73b91061988fa6a` | `train/neg/12319_1.txt` (label=0) | `train/neg/7346_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `886a4b01e6e6fc2c437e4cd0e1b13893d979e7bba199ea30041412828c2fc6bc` | `train/neg/7869_2.txt` (label=0) | `train/neg/4983_2.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `b0b7cd4db162e810e0df8eaefb2bb5d9d205b9d42ae55ae986af82cda471fbfe` | `train/neg/4988_2.txt` (label=0) | `train/neg/7874_2.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `b402625ae82d5b61639b2202f6c7bd6c094cbae178a904de254e54e49db9880f` | `train/pos/9319_8.txt` (label=1) | `train/pos/6069_8.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `b72f81eec0b8e0817e7e148929567ac821bf2478a87724cef0fbb2ba7e31f669` | `train/neg/5583_4.txt` (label=0) | `train/neg/2716_4.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `be071c07c71b4bd66e97fa62fb52e026ab27e0e1a1a4f6bf96e2685e7330640e` | `train/neg/216_4.txt` (label=0) | `train/neg/215_4.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `cbcf3ec5fb5089e43de33c78b891ad45d00f08fa3ea603399559dee6c3335f13` | `train/neg/10108_1.txt` (label=0) | `train/neg/10089_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `d71ec64fc79fb043ca043b55f1005ae56e8a6e7fb32453244d4db6e8ace7ae5c` | `train/pos/4267_8.txt` (label=1) | `train/pos/4263_8.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `decfb403379023760711186f35469b825bc0f370e85ea7306c5c919b38ea8f6a` | `train/pos/972_9.txt` (label=1) | `train/pos/973_9.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `f1b48a58b6d2582fb86e7fe73095216d5ba152be4ae5d14200fa43d17e961a87` | `train/neg/987_4.txt` (label=0) | `train/neg/4104_4.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `f1e1a5a801707926a1d9f215704e2a7f3e7d839304b5e6fa32475acdb5ba5ddd` | `train/neg/5952_4.txt` (label=0) | `train/neg/11974_4.txt` (label=0)<br>`train/neg/5550_4.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `f625ce09af9816f2e64d7a2291991a9e246603d543a3c65de65975140e5e9fd8` | `train/neg/7873_1.txt` (label=0) | `train/neg/4987_1.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `f9719ec3987629562d9b8081229cbc7ae6d5ecedbc79d5630f3e67fbb804e357` | `train/neg/11815_3.txt` (label=0) | `train/neg/11209_3.txt` (label=0) | - | Byte-for-byte identical |
| train / validation | `fb93d800cfcfa6e679c50c53ed2de1e2ac3f81e3358d52a89af665aff1180aa3` | `train/pos/9661_8.txt` (label=1) | `train/pos/3312_8.txt` (label=1) | - | Byte-for-byte identical |
| train / validation | `ff8ca4ac65e3b9bba1826a203f7e825bfb88741d798934149cb2922f8b363849` | `train/neg/6645_1.txt` (label=0) | `train/neg/5088_1.txt` (label=0) | - | Byte-for-byte identical |
| train / test | `032e389b75ac2a7532d0288c717a675f0303ca9525e4e1cd2ad0bf5b3f711f2e` | `train/neg/9452_1.txt` (label=0) | - | `test/neg/12310_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `03e64649da1abdcf4cd6233533f0b771144adb3c54211591af7e94f1fb25ca32` | `train/neg/12289_1.txt` (label=0) | - | `test/neg/60_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `097624f02b93350de8bddf01ce7825b450ab7c92968be77f61c27fb86f07f102` | `train/neg/121_4.txt` (label=0) | - | `test/neg/1846_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `09a146dd0b42a8e9dbf6ec004e2c2d96ff63219db72c2ef8df2ed5bb0588fa4a` | `train/neg/10750_3.txt` (label=0) | - | `test/neg/4903_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `0a7b9514f14e9dd6297fbfa4f5b37e33b56747b483dd95ce06df26a72fe44184` | `train/neg/9459_1.txt` (label=0) | - | `test/neg/12317_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `0a8a8468e8866f975f173234fe98065e0c6251ad7fed0873437e3ef3510d3820` | `train/neg/9447_1.txt` (label=0) | - | `test/neg/12305_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `0a95827c5ab7872f7e4f69f5e77104d1cbd4eb6d18e7a0ff29f33f19a2c45327` | `train/neg/12296_3.txt` (label=0) | - | `test/neg/67_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `0ba6a9d04a7ca6f94f65b7686d4489cfa4ad4345fd397b472b66ecc7a93b353e` | `train/neg/12290_1.txt` (label=0) | - | `test/neg/61_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `0c8400dc3d320f1447599c45d904a17fc8c9819bdc7125a5b08fd5307f3953c6` | `train/neg/120_1.txt` (label=0) | - | `test/neg/1845_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `0f909b9c48ce4fd08364fe8dff206db56797cca1310a6001d2332120523cff83` | `train/neg/10206_3.txt` (label=0) | - | `test/neg/1540_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `13ad7c6a8c927811a29e4982ce7c2c4cad6fbb615c685827a46aa9f7ffe745b9` | `train/pos/2404_8.txt` (label=1) | - | `test/pos/9864_8.txt` (label=1) | Byte-for-byte identical |
| train / test | `18dd172a0c48e203ba923168a53182e79931c8593ef53585507c81e86ca6b93c` | `train/neg/1655_2.txt` (label=0) | - | `test/neg/12043_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `24107cbd7b0ae0bb6465bcc677f6cc077a92b37e92399f4c9aaaf8bd79c70ed0` | `train/pos/2405_10.txt` (label=1) | - | `test/pos/9865_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `25d70157cd3b34a4b90c4e0e3b9661d94f891f3d0071a7ac7120da704b487f15` | `train/neg/9472_1.txt` (label=0) | - | `test/neg/12330_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `26aa4fe0e76ca641cc64672eb224d70cb8081394611f7ef56e165796424d199d` | `train/neg/10393_4.txt` (label=0) | - | `test/neg/3885_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `29395ad2c36c2181dd301a6b5a219d36d6f6e9280fc92c8531df53c5828d6bdd` | `train/pos/8446_10.txt` (label=1) | - | `test/pos/9262_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `2974a9d524a9387785daa5e0b68a78f870d6286d1d299295b130ac2952fa9010` | `train/neg/10203_2.txt` (label=0) | - | `test/neg/1537_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `2a58657f5995921a9e63a03e0931d53f1a3bc38869ea569c5feabb8c740ebe96` | `train/pos/8440_8.txt` (label=1) | - | `test/pos/9256_8.txt` (label=1) | Byte-for-byte identical |
| train / test | `2a86c967557b99b44b6468b58975fef16d9129bb3641dde91be3124eba3cca4c` | `train/pos/2403_10.txt` (label=1) | - | `test/pos/9863_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `2ca2bda0ada50b89b08c39de6d501789ade878378a5e6ec2d690b5f6c3c28b99` | `train/neg/12094_3.txt` (label=0) | - | `test/neg/2884_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `2d44eff3bb0b258901d7b077472d33369e84663097dd565c32d42dadc4f247af` | `train/pos/7228_9.txt` (label=1) | - | `test/pos/9610_9.txt` (label=1) | Byte-for-byte identical |
| train / test | `326387b6d4198de2dfe5c69c4ae2d902d6bb7c5b03e75cee4c702cb74c78bc65` | `train/neg/10205_2.txt` (label=0) | - | `test/neg/1539_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `32731b38994b5efe64b45b6504adcdf75117b87ed77a59da712307b8880a9ec8` | `train/neg/9462_1.txt` (label=0) | - | `test/neg/12320_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `375c1d62feac9a2ccb597009abaa7b37321c8a58b6effc28bdc953880dfedb12` | `train/neg/9471_1.txt` (label=0) | - | `test/neg/12329_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `378031daf8209cab1d9787b7ff08149c736cf2c1668da3b1c1e3c5b4204092d2` | `train/pos/8441_10.txt` (label=1) | - | `test/pos/9257_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `37f6cba7d486cc04947241e5969c03d01f0c136e4375a91f7c915fb5127ea164` | `train/pos/308_8.txt` (label=1) | - | `test/pos/6910_8.txt` (label=1) | Byte-for-byte identical |
| train / test | `38cd001364a90f6566aa46f842e992bfacb34a7e78cd89ac973c7ac860a9df95` | `train/neg/6797_1.txt` (label=0) | - | `test/neg/7465_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `39725d38ded0ccd68315f82f299e2b8e2d362fa63598c49f5ac4378d09b8cf2d` | `train/neg/9467_1.txt` (label=0) | - | `test/neg/12325_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `3a5f9b74d00fcee05cfb663a4e81a35090012d14db6e47806ab48d1b9528b4ad` | `train/pos/8435_10.txt` (label=1) | - | `test/pos/9251_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `3b96cd0af7ab05b439ea58225a484785ef5fa76657c1f7ebd63882df1e02f1ff` | `train/neg/10204_2.txt` (label=0) | - | `test/neg/1538_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `3baff315bd14a2bc859015f3aa46691554626f62de5365302776986d8764e647` | `train/neg/9448_1.txt` (label=0) | - | `test/neg/12306_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `3f6f5b760fb2640aa5dfb878169572f2e23903a5393efe8876797a261b8fc1d9` | `train/neg/10198_4.txt` (label=0) | - | `test/neg/1532_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `48e6f9f3f1d44f5de7efb577afbe05e37c04191b1437c7c4d7e85d54b1a99163` | `train/neg/9444_3.txt` (label=0) | - | `test/neg/12302_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `4a795a3fc51320417d5073c37a0f73cb01edc95dbd384eb851ef1b40c4fcd957` | `train/neg/3412_1.txt` (label=0) | - | `test/neg/4693_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `4b2d6f5e37b21d54cdd553aec1d837f0fb27eb2c2cbc4e0067d5ba974972a07c` | `train/neg/12089_4.txt` (label=0) | - | `test/neg/2879_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `4f77022c9ea9ef2f8b78565f75ade9a23a03a2e6298bc9e64812b044be05514a` | `train/pos/8439_10.txt` (label=1) | - | `test/pos/9255_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `4f80000250901a0f1a12658cd4bfc3759f47d252d1cc7825fd56ce54e5e0a0ca` | `train/neg/9464_1.txt` (label=0) | - | `test/neg/12322_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `530e663b952423bd17635c507b8ff2e82be77f56a93572f8e6454f9aa028f3ad` | `train/neg/12298_4.txt` (label=0) | - | `test/neg/69_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `54e38afc73080ec6e5f270a7e056c9c92940cd2baec3b176bcad91870c07c10c` | `train/pos/310_7.txt` (label=1) | - | `test/pos/6912_7.txt` (label=1) | Byte-for-byte identical |
| train / test | `5810d0e97a5435b39b4dd932b1fb7162e8d43d40cf8a092b1c82bdf581e930da` | `train/neg/12300_4.txt` (label=0) | - | `test/neg/71_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `5d46101fa648c5c1ba15336cc4ae88801e1fbd94dd87a49689e232d99254a0f8` | `train/neg/4518_2.txt` (label=0) | - | `test/neg/15_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `5ee254ca5c3397de5375e85c32b58d6f4e08f9c3969800e5a9192e8b4469169c` | `train/pos/7227_7.txt` (label=1) | - | `test/pos/9609_7.txt` (label=1) | Byte-for-byte identical |
| train / test | `605ad823bf9313ac51919b2f9866a721680ed4f51510088d166b0ffc62832ed5` | `train/neg/12093_3.txt` (label=0) | - | `test/neg/2883_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `619dbfbb62959c5850800f6a6dd5f24d3a7c65e07eb005cae87ac2524793df83` | `train/neg/12293_3.txt` (label=0) | - | `test/neg/64_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `655a99997c6383b77f0d7ff76b634c814610d74bf794c1c0bfde32d4f8732a34` | `train/neg/10207_2.txt` (label=0) | - | `test/neg/1541_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `65b12396aba811662b7946a36ebeae619b4f5f5f32d4d26448df9a1c125a2ab3` | `train/neg/9457_1.txt` (label=0) | - | `test/neg/12315_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `66fc1ed075bc334e305b6e66b8ee46de459fb8373399c8ec5479fca89b9bdc65` | `train/neg/5440_2.txt` (label=0) | - | `test/neg/1790_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `6a6fb65366faa871cb3c0b727fe1256a22c447e2c8928c17afed7667b89e9bc6` | `train/pos/10643_8.txt` (label=1) | - | `test/pos/12159_8.txt` (label=1) | Byte-for-byte identical |
| train / test | `6ddaf44e899cafa289f961dd86484abd7b1dd94881f84e8f0c38703bf6d91d14` | `train/neg/12091_3.txt` (label=0) | - | `test/neg/2881_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `719884aa7b1220578f91b22b581ba76acaf82bcf49e4a664d7044aba4874fd0a` | `train/neg/9470_1.txt` (label=0) | - | `test/neg/12328_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `72ab29789036dd3b85ada21db91c1a54d6e536552fb64bbe5acad6b2c116feb4` | `train/pos/8442_9.txt` (label=1) | - | `test/pos/9258_9.txt` (label=1) | Byte-for-byte identical |
| train / test | `72ad69f0f1a342f9508e584179d2f2c63127ae83457a11ad4278198f47876a7b` | `train/pos/309_9.txt` (label=1) | - | `test/pos/6911_9.txt` (label=1) | Byte-for-byte identical |
| train / test | `72bf6d671e55ad7e06ff70409ed97629115024615ef9f73ecb907cdedb43f3c5` | `train/neg/12294_2.txt` (label=0) | - | `test/neg/65_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `74e5f2493cf79e8bbe84c14a6ddd87cbcde34fbdab8bfa3e1023115d36f6d5ed` | `train/neg/10199_4.txt` (label=0) | - | `test/neg/1533_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `7779e714c5763716552f73042d12d68eefec4ad08c25ed20498bfec7db8590bf` | `train/neg/9460_1.txt` (label=0) | - | `test/neg/12318_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `77a1c8c41b477ecb3bfb76b0e577f3a3b688bcee0b3bd9d1bdafb856243c3440` | `train/neg/9454_1.txt` (label=0) | - | `test/neg/12312_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `7b3bc49e6c977e7c22e44c878a4ed18f8823dd4dd6409c13d75818684b24f6d4` | `train/neg/5442_3.txt` (label=0) | - | `test/neg/1792_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `7d4d41c350cd01760a25b0e8fdc634b51462c2918cd3322b81f8a6ff7843581d` | `train/neg/8091_4.txt` (label=0) | - | `test/neg/9507_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `7e33f210a4bb94c0932f90e74b33ecbc922563c09b3d1f1aea23c0edc968e029` | `train/neg/6261_2.txt` (label=0) | - | `test/neg/6402_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `854fef0bfde723665297cce276705780db2fc5e1e0151048166ee1529f3bd768` | `train/neg/9449_3.txt` (label=0) | - | `test/neg/12307_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `85b15059edf2dcaf91a00dd55635563ffe48dfd1b3e5552543af5c4bd0343f34` | `train/neg/9463_1.txt` (label=0) | - | `test/neg/12321_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `88b0e4f703a7d1f93ee4fdf0936a6cba30391a4aaf078c567b03492f4974bd49` | `train/neg/9398_4.txt` (label=0) | - | `test/neg/9644_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `8a200c9c3b7f34645f39e565487f36610f301c10d28011501b4a1e79a812a3f1` | `train/neg/12292_1.txt` (label=0) | - | `test/neg/63_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `8e8bf43a5956164240b576a64a0ddfdccac8f99a83693b35624ddabd9441791b` | `train/neg/10200_4.txt` (label=0) | - | `test/neg/1534_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `8fd638f147e390439d403e28e1c1c7571745c86a49e02b5934cd0ccc1c37a4fb` | `train/neg/12295_3.txt` (label=0) | - | `test/neg/66_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `9075a78e387ec4d6df72c59ac9ef1d573e647328a82474bf12cd151e91f119d3` | `train/neg/9469_1.txt` (label=0) | - | `test/neg/12327_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `9a4ecb0aa423af4b910044e056082fbb31e92bff19b270f98879123136280bfc` | `train/neg/5441_1.txt` (label=0) | - | `test/neg/1791_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `a1b2a237bcec803c18c96e59be25a3f4bca0611f5a441556c83c6b74106f4ff8` | `train/neg/10210_3.txt` (label=0) | - | `test/neg/1544_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `a45b84e3288556c228eab16ca4fd33bbdc7d999cb4c722f3bd17db06b969686b` | `train/neg/5459_2.txt` (label=0) | - | `test/neg/2844_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `a4c7b451791b0da485c7d275b1fcfcc20f1368930e86b0755e48f1e78d7720e5` | `train/neg/9461_1.txt` (label=0) | - | `test/neg/12319_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `a80ff7d317a3d0dc4c4c76d0e37c48399031b1640d8712395b2484e78764bf2c` | `train/neg/2164_2.txt` (label=0) | - | `test/neg/2557_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `aa8e197182bce1c1c39132a1a1d52fe65277ee9c70df7f36f897a58c5fc663b2` | `train/neg/2165_4.txt` (label=0) | - | `test/neg/2558_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `ab8f3a74f6af5fee9c14dafb9dcb6e1dc122452376712eca925c5d01addd08d2` | `train/neg/6796_1.txt` (label=0) | - | `test/neg/7464_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `abba3b0a48bc1cfe280ba1a7bd01a413e2a35e8053195452b91ff9c4a8d63979` | `train/neg/12291_2.txt` (label=0) | - | `test/neg/62_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `b0153fc29c799579cee509ee5c657304c2b3b599d68870db424ee25f386b7058` | `train/neg/12299_1.txt` (label=0) | - | `test/neg/70_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `b5798548b2ac37f7f49c055896b654eb64d136a51240ebebd6bc8985c1ba23eb` | `train/neg/9468_1.txt` (label=0) | - | `test/neg/12326_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `b7a26ca09b708f60cbb1ec994bffe2df9fae60bbf4fe3aedb1f3e19f55fcbf7d` | `train/neg/9466_2.txt` (label=0) | - | `test/neg/12324_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `b8c4bcb3eb1c1061e66b72ab83f7f6efe22666fb034034f1d81f882fc57cf912` | `train/neg/6260_1.txt` (label=0) | - | `test/neg/6401_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `be8bc79cc1eccd004c3f95acdc84a4064e3170e50f7bc1ea635ec590e634db13` | `train/pos/8437_10.txt` (label=1) | - | `test/pos/9253_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `c195e0e98d723e1a81f844e7130b37e1a4cf3ca6bec3282374929ea89720fb5d` | `train/neg/2534_3.txt` (label=0) | - | `test/neg/2486_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `c282ea5cabfcce859d00581ffa534a393581f58cd4dfd502abf59b525df0cb67` | `train/pos/8445_10.txt` (label=1) | - | `test/pos/9261_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `c93088c0a82617a456392139ce83f842e2e5e522b6492774cb833b034324d0d6` | `train/neg/9456_1.txt` (label=0) | - | `test/neg/12314_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `cbcc0a4a4c6f76dbc4899c1949d25aebf0e4d0bdc7eb968ed6c1a8062ab7a657` | `train/neg/9446_1.txt` (label=0) | - | `test/neg/12304_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `ccb577f6811791f9290fd9cc533debf8a80dc56bee199eb65e53b7f94697a65c` | `train/neg/8090_3.txt` (label=0) | - | `test/neg/9506_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `cccea0d3733f26838cefe3b7a48c7b6e6b828944db90a49284def54d219febf7` | `train/neg/9450_1.txt` (label=0) | - | `test/neg/12308_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `d171a3c206ef2c07ab90faa32a31d41819d149cb4f261bf3202fed84a6e94c8f` | `train/neg/9455_1.txt` (label=0) | - | `test/neg/12313_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `df5ac32f50b74da33945899f5afc87f2fe7fd379ed19cb3d595c6c3db43fdd72` | `train/neg/10394_3.txt` (label=0) | - | `test/neg/3886_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `e16b09dbe915d345f3866fc026b6c5bd4d33b2be87d498ee91be1988d0735414` | `train/neg/10395_2.txt` (label=0) | - | `test/neg/3887_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `e1902eca4ca8d388c467a26e9bb2e9de094c8ba1e1ade4b43ba80b4466a7f43f` | `train/neg/9453_1.txt` (label=0) | - | `test/neg/12311_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `e53cff34b58e7110ca2749bca491eef61075a1f9e1ee304fcab3283a257278ad` | `train/neg/10208_3.txt` (label=0) | - | `test/neg/1542_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `e7335c6adeb4f8889c54754cc73ab61b96b5c74ac4aa6b942baee33d2708f716` | `train/pos/8438_10.txt` (label=1) | - | `test/pos/9254_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `e75470db38b3caceef9df8a99d8cec4ba24120ad293afe907f9f7ff5c5a7abc4` | `train/neg/5443_1.txt` (label=0) | - | `test/neg/1793_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `eaeaaef40929c853584bba34bc336573f729ae395024da0328173e2cbee774ee` | `train/neg/12092_3.txt` (label=0) | - | `test/neg/2882_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `eb5e09176b3797997fe5e5077dbfef19d5717c3ff340e64a1227e7a4927a67cf` | `train/neg/2166_4.txt` (label=0) | - | `test/neg/2559_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `f0e1c441af84038e241644ac370acf297ce0b1625de62630a259e4107c458d32` | `train/pos/8434_9.txt` (label=1) | - | `test/pos/9250_9.txt` (label=1) | Byte-for-byte identical |
| train / test | `f9848e0093002b5c8fc6e8b0e90a90fc40a565d5e32facffa1bd095f07e86cb7` | `train/neg/9458_1.txt` (label=0) | - | `test/neg/12316_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `fa2265fcce33d0b341609cf47c966f17aac9e7948a3d0d4c6f9b00b54ef35c69` | `train/neg/770_3.txt` (label=0) | - | `test/neg/2665_3.txt` (label=0) | Byte-for-byte identical |
| train / test | `fad988b5ba73250b2655d5dd1e41706c697635e2a18b18a2c27a14aaf4be8153` | `train/neg/3413_4.txt` (label=0) | - | `test/neg/4694_4.txt` (label=0) | Byte-for-byte identical |
| train / test | `fb1f58e8ee4c7e84c49d126334c37f5135923044699cd3ea3f2644ddb47fb089` | `train/neg/12088_2.txt` (label=0) | - | `test/neg/2878_2.txt` (label=0) | Byte-for-byte identical |
| train / test | `fbcfb5bbaa5284c7538b4239ece86476c8bed6f444a7e25f504eb904bc267f20` | `train/pos/8436_10.txt` (label=1) | - | `test/pos/9252_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `fc38b5d13d4c0feb4c0a21d7550a26e4e51e6f639ca3a7292181c1946551365c` | `train/pos/8444_10.txt` (label=1) | - | `test/pos/9260_10.txt` (label=1) | Byte-for-byte identical |
| train / test | `fc48056961326399c0a89aa70e3b49cd0b5cdfd129e796e22967c2acd93bfa09` | `train/neg/9445_1.txt` (label=0) | - | `test/neg/12303_1.txt` (label=0) | Byte-for-byte identical |
| train / test | `ffd82d0c56d56cf853bb66681b7f1b72d7ce37d4e13888354a101c9b235ea381` | `train/pos/311_9.txt` (label=1) | - | `test/pos/6913_9.txt` (label=1) | Byte-for-byte identical |
| validation / test | `011e8663d9447492d45d42f75d15e1d5cc0785417a0688d1d7964f10f976e1a6` | - | `train/neg/9537_1.txt` (label=0) | `test/neg/8946_1.txt` (label=0) | Byte-for-byte identical |
| validation / test | `0da909f235f7076bf2e7ecec6ebcb5b436a4888f7285e8a6edf0ff379c737712` | - | `train/neg/1870_1.txt` (label=0) | `test/neg/8173_1.txt` (label=0) | Byte-for-byte identical |
| validation / test | `19c4cbbd01b4f0f9aea0cefe5a2b0b016c7b878f4423ec552f27e5f8225a2377` | - | `train/neg/10209_3.txt` (label=0) | `test/neg/1543_3.txt` (label=0) | Byte-for-byte identical |
| validation / test | `38a40de1d0e544d59675e04e697dcaa7a957a47b2bf321bcef876daaf9f5842c` | - | `train/pos/8443_9.txt` (label=1) | `test/pos/9259_9.txt` (label=1) | Byte-for-byte identical |
| validation / test | `4be7c9e753caadff2678cabc4a4378855c01ef8979d8f531740053a33193daef` | - | `train/neg/10201_3.txt` (label=0) | `test/neg/1535_3.txt` (label=0) | Byte-for-byte identical |
| validation / test | `513430c5c9a417662aacac576af3569ff05cd828ad03dd71d3216be064199f05` | - | `train/neg/1656_4.txt` (label=0) | `test/neg/12044_4.txt` (label=0) | Byte-for-byte identical |
| validation / test | `51da85f392f462d153019d9bc8823f54860e225041e4b01a730d74bd88f148d7` | - | `train/neg/12090_3.txt` (label=0) | `test/neg/2880_3.txt` (label=0) | Byte-for-byte identical |
| validation / test | `57cbed930a2f2eabd7708a8a849483d44fd05bf11f3521741c0ef22d7c3a3e79` | - | `train/neg/12301_4.txt` (label=0) | `test/neg/72_4.txt` (label=0) | Byte-for-byte identical |
| validation / test | `77b4739cfd771e530e776a8623304fcf33bd62911433c721afb3e990a2498118` | - | `train/neg/12095_3.txt` (label=0) | `test/neg/2885_3.txt` (label=0) | Byte-for-byte identical |
| validation / test | `8fc3c22ffc7b2ac49e6e5dd0546b0148bc0917f489d20f75cf41efea80322869` | - | `train/pos/8447_8.txt` (label=1) | `test/pos/9263_8.txt` (label=1) | Byte-for-byte identical |
| validation / test | `9502cae9c3a96b90531b9129283f0b02845145a29cd7919817d2b96564616af7` | - | `train/pos/8448_10.txt` (label=1) | `test/pos/9264_10.txt` (label=1) | Byte-for-byte identical |
| validation / test | `aa1f1c5928e002d5d62aadd65c056659f249ae33a6bd7ff067393b4f38df08fe` | - | `train/neg/9399_4.txt` (label=0) | `test/neg/9645_4.txt` (label=0) | Byte-for-byte identical |
| validation / test | `bac201570cbff29597e12c7f5c559b609a42ac3d0527e81d6b8feb6df8b1df17` | - | `train/neg/12302_4.txt` (label=0) | `test/neg/73_4.txt` (label=0) | Byte-for-byte identical |
| validation / test | `c41f82611a71e2331b1c7d80be913df926faf232e0af9a7097f2a6520d703ca4` | - | `train/neg/9465_1.txt` (label=0) | `test/neg/12323_1.txt` (label=0) | Byte-for-byte identical |
| validation / test | `c568a5cf8cc9dfc75d4cc89762bf4ad2969d9f829f92d687e83d50f1668b21ef` | - | `train/pos/7226_8.txt` (label=1) | `test/pos/9608_8.txt` (label=1) | Byte-for-byte identical |
| validation / test | `cb0704ac35e36662f2c5d3baef3e48006771157204cf6779d6c685fdcabcc828` | - | `train/neg/10202_3.txt` (label=0) | `test/neg/1536_3.txt` (label=0) | Byte-for-byte identical |
| validation / test | `cd50e6bf31435628bedd812808cb24b08ca64292f5c19fc90a697c14ac7b0ba1` | - | `train/neg/12297_1.txt` (label=0) | `test/neg/68_1.txt` (label=0) | Byte-for-byte identical |
| validation / test | `d1a2c8b3486b10b163dcc85ca25fd1c8ee7d2a01cfb9ed72d7ed4af9d71dc66c` | - | `train/neg/9451_1.txt` (label=0) | `test/neg/12309_1.txt` (label=0) | Byte-for-byte identical |
| validation / test | `e68f78d2c2d3cc4d0fb75079759b5be2a0193b7b07e4fb6ae38e15c087fecf0a` | - | `train/pos/8449_9.txt` (label=1) | `test/pos/9265_9.txt` (label=1) | Byte-for-byte identical |
| validation / test | `f3ce8b85ca1defa01b83648daad2b24eca3302b6ab465b41863437fa840e5e2c` | - | `train/pos/7225_10.txt` (label=1) | `test/pos/9607_10.txt` (label=1) | Byte-for-byte identical |

## Policy comparison

### A. Preserve the existing split and disclose the limitation

This preserves the already-used examples and Model A artifacts, so the current comparison remains directly paired. However, exact review bodies cross into the validation and official test sets. Report metrics as potentially optimistic and explicitly disclose the contamination; do not describe the held-out test as fully leakage-free.

### B. Deduplicate/group before creating a new split

- Deduplicating all labeled source files by the stated normalized-text key leaves **24,904 unique texts** in original train, **24,801** in official test, and **49,582 unique labeled texts total** across both source partitions.
- The requested 20,000 / 5,000 / 25,000 totals sum to 50,000. They **cannot all be maintained without reuse** from the deduplicated corpus because only 49,582 unique texts remain. Keeping a distinct 25,000-review official test also cannot be done from its deduplicated source if it has fewer than 25,000 unique texts.
- All normalized text groups have a single label, so group-level stratification can preserve class proportions closely. The globally deduplicated corpus has 24,698 negative and 24,884 positive texts (186 more positive groups); exact 50/50 balance over every unique group would require dropping 186 positive groups or changing the corpus. Exact 20,000/5,000/25,000 split sizes remain impossible without reuse because only 49,582 unique texts remain. Keeping the source partition also leaves 24,904 unique original-train texts (96 short of the requested 25,000 train+validation) and 24,801 unique official-test texts (199 short of 25,000).
- A new split changes the training, validation, and test examples. Existing Model A `best_model.pth`, tokenizer, training history, and test metrics would no longer represent the new experiment; Model A must be retrained and reevaluated. Any Model B checkpoint, history, configuration, or metrics from the old split would likewise be invalid for the new comparison. No Model B artifacts currently exist in this repository.
- Both models must be retrained and evaluated on the same new grouped split for a scientifically comparable Model A vs. Model B result.

## Recommendation

For a leakage-controlled scientific comparison, prefer **Policy B**, provided the project can accept revised split sizes and retraining both models. The fixed 20,000/5,000/25,000 split and the no-retrain Model A rule are incompatible with eliminating the measured duplicates while retaining 50,000 distinct reviews. If those preservation constraints are binding, Policy A is the only option that preserves the existing experiment; disclose the limitation and characterize the held-out metrics as leakage-affected. **No policy has been implemented.**
