# Changelog

## Unreleased

### Features

* add a JAX/Equinox transaction core for discrete-event observing campaigns, with shared event ordering, atomic admission, resource accounting and semantic random keys; host and compiled executors; typed records with audiences, release times and revisions; a snapshot-bound host request/result protocol; structural pre-flight checks; a generated validation battery; and a conformance corpus with hand histories. Run the acceptance gate with `pytest tests/test_slice1_gate.py`.

## [0.2.0](https://github.com/CoreySpohn/spaceodyssey/compare/v0.1.0...v0.2.0) (2026-11-07)


### ⚠ BREAKING CHANGES

* **core:** remove the certificate stack that registered into the removed registry
* **core:** remove the spec, registry and wiring mission loop in favor of the core and codec
* **core:** observer fold and fold_summary take key=; keyed, padding-safe refit over refit_rows

### Features

* **adapters:** first-summary-v1 profile table, fixture pins and profile-table test ([0f132f8](https://github.com/CoreySpohn/spaceodyssey/commit/0f132f83405ae57ec1589e92cba8b9c72caeee97))
* **adapters:** hand-built skyscapes scene, prior world source and truth-side diagnosis ([6e815ef](https://github.com/CoreySpohn/spaceodyssey/commit/6e815ef32d3f1209556dd1489294af34b446db59))
* **adapters:** observing context with absolute time, science intervals and keepout certification ([fe5af1d](https://github.com/CoreySpohn/spaceodyssey/commit/fe5af1dc6fab160fdef15c0078248592d2d32043))
* **adapters:** OFTI host-loop refit with a physical fixed-mass model declaration ([b795d2c](https://github.com/CoreySpohn/spaceodyssey/commit/b795d2c16bcc6796fade01b4e8a2b104956eaa67))
* **adapters:** replay planit plan tables through PlanTablePolicy ([5886cbe](https://github.com/CoreySpohn/spaceodyssey/commit/5886cbe2fe586d3930959a31f326578932dadd39))
* **adapters:** summary radiometry through jaxedith with time-integrated counts and variance components ([581dc7e](https://github.com/CoreySpohn/spaceodyssey/commit/581dc7e2b9f72fb18bf5b5f6c6c3c96e0b6ea6f6))
* **adapters:** summary reporting law, record kind and calibration id ([b31cf73](https://github.com/CoreySpohn/spaceodyssey/commit/b31cf737292807f4034aadd22ee53684eb7c2ec0))
* **codec:** add canonical study documents with a generic field walk ([322645b](https://github.com/CoreySpohn/spaceodyssey/commit/322645b9368c84002d39d637fdbff02c33caa814))
* **codec:** load study documents with exact types, dtypes and pointed errors ([d5201d4](https://github.com/CoreySpohn/spaceodyssey/commit/d5201d42c9c3eac0c13723d25469d7f59c4851b6))
* **codec:** rebuild the run record's refit boundary for the JSON pre-flight ([7503f01](https://github.com/CoreySpohn/spaceodyssey/commit/7503f01dde3e97a8adb7fe15d210cc506076eeaf))
* **codec:** report all thirteen pre-flight codes on a JSON document ([c2dcbc9](https://github.com/CoreySpohn/spaceodyssey/commit/c2dcbc935f0fa592abf9ef48da231ea978083fc2))
* **codec:** resolve kinds through KindResolver and migrate older kind versions ([813a59a](https://github.com/CoreySpohn/spaceodyssey/commit/813a59a50c6009ce9ab6c484bddc5bc3d631b53b))
* **codec:** run document with service and refit bindings, registry, file references, callable refusal, no_switch ([899f93d](https://github.com/CoreySpohn/spaceodyssey/commit/899f93de5a4c174b20bce0c0d3f70d3c39d629f6))
* **codec:** the JSON pre-flight takes an upstream document, so D004 is reached at the declared hash ([f9fa76c](https://github.com/CoreySpohn/spaceodyssey/commit/f9fa76c7b8ab41da5b803e3c7c588a2ca6e5b90e))
* **core:** canonical history, battery and fold counter read the belief layout ([b95b01f](https://github.com/CoreySpohn/spaceodyssey/commit/b95b01fa35c29c1e3f7308dcc5e4a1185dbd5efb))
* **core:** keyed replay with refresh and roots on run results ([47c7d9a](https://github.com/CoreySpohn/spaceodyssey/commit/47c7d9a0d900666ac680a3dd5d2bf3fbc995ab61))
* **core:** kit checks delayed corrections and derived summaries; refits keyed by refit_key ([c6bae7e](https://github.com/CoreySpohn/spaceodyssey/commit/c6bae7e2f3a7e58b981858e092ff92b6d47e579f))
* **core:** numbered Agenda levels as nested effect masks intersected with phase masks ([c0c0929](https://github.com/CoreySpohn/spaceodyssey/commit/c0c0929dcdc240c468cb2d12b787b981582161de))
* **core:** observer belief layouts; snapshot_fingerprint takes the layout ([8c1ae0b](https://github.com/CoreySpohn/spaceodyssey/commit/8c1ae0b3906f3f2f82fa6a5328f729f105b0ea70))
* **core:** observer fold and fold_summary take key=; keyed, padding-safe refit over refit_rows ([6cf687f](https://github.com/CoreySpohn/spaceodyssey/commit/6cf687fa9ce4811b17c8270edcd848201318d45b))
* **core:** observer kit checks of overlap weights and zero-weight rows for observers that read priors ([119079f](https://github.com/CoreySpohn/spaceodyssey/commit/119079f3c33e9e1d7a9e1f32440021a484a0d0e6))
* **core:** observer kit with calibration, single-fold, promotion-parity and margin checks ([764fd1a](https://github.com/CoreySpohn/spaceodyssey/commit/764fd1a515c01b0519da7257b0dae5859b0bbe25))
* **core:** observer refit, summary and install methods; folds take their inference key ([0d94aa5](https://github.com/CoreySpohn/spaceodyssey/commit/0d94aa5d81fb7cf42ea476f8608eecd3a02d55de))
* **core:** pedigree column, revision sets, derived kinds and producer-local identity ([c5412fe](https://github.com/CoreySpohn/spaceodyssey/commit/c5412fe513627e34e0b0d1e5860b69959a23bbae))
* **core:** pre-flight context and a validate hook for component findings ([0c3dbf4](https://github.com/CoreySpohn/spaceodyssey/commit/0c3dbf41db1d18bfd74789f829ac8c5c2599a940))
* **core:** refresh protocol with checkpoints, hold and install, masked and host-loop boundaries ([6e5883d](https://github.com/CoreySpohn/spaceodyssey/commit/6e5883d1c241d2553f55c9ff9ffbbf41eb8d8e54))
* **core:** remove the certificate stack that registered into the removed registry ([15ed8f4](https://github.com/CoreySpohn/spaceodyssey/commit/15ed8f4a156048d597069244cf091df498ab5be1))
* **core:** remove the spec, registry and wiring mission loop in favor of the core and codec ([8e77dda](https://github.com/CoreySpohn/spaceodyssey/commit/8e77ddacaeb6482d9e9afea5939c96446a00d340))
* **core:** require layouts in history and answer(); harden layout checks ([ab85828](https://github.com/CoreySpohn/spaceodyssey/commit/ab85828e70151c54642e5a1cecc060158434a528))
* **core:** shift_epoch runs on either executor and refuses a shift that is not positive ([5d0f50c](https://github.com/CoreySpohn/spaceodyssey/commit/5d0f50c693113e786314db302722de952d0775db))
* **core:** slice 3a capacities, conditional leaf tables, optional codec fields, key_version ([f155d0b](https://github.com/CoreySpohn/spaceodyssey/commit/f155d0b8c2a4e5db1a0cd5e77e3750eaf7e1c04b))
* **plugins:** discover kinds from spaceodyssey entry points ([13f6f3a](https://github.com/CoreySpohn/spaceodyssey/commit/13f6f3a690079122301d81458ba970233d6ba1a4))
* **studies:** export a precursor table from an upstream run, at the latest revision released to the audience ([ab01d0b](https://github.com/CoreySpohn/spaceodyssey/commit/ab01d0ba852e7a524bd475c1da5b55b8e75c145b))
* **studies:** OFTI stand-in refit of physical summary rows with particle observer, fixed plan and scripted rows ([925e8f8](https://github.com/CoreySpohn/spaceodyssey/commit/925e8f8c2e6c360e74a8b73c62058b247d14672e))
* **studies:** per-audience overlap rule, prior observer and audience-prior refit boundary with findings D008 to D010 ([ec013c3](https://github.com/CoreySpohn/spaceodyssey/commit/ec013c39af796940cf4c1e335225c4969b13dc5d))
* **studies:** pre-mission table, source identities and importer with findings D004 to D007 ([99f36b9](https://github.com/CoreySpohn/spaceodyssey/commit/99f36b9c599397d82eca4f458366f70a81700eaa))
* **studies:** survey 14, a prescribed phased mission with a no-switch reduction ([a2a84c0](https://github.com/CoreySpohn/spaceodyssey/commit/a2a84c0255b9254ceb0883ea94fecc71d24867ff))
* **studies:** survey 18, a star-level precursor that chooses between two plans ([a90a85d](https://github.com/CoreySpohn/spaceodyssey/commit/a90a85d24f53c489cdf5a6d94a0f36d35b726fca))
* **studies:** survey 2, an adaptive revisit search with calibration and an OFTI variant ([e82832e](https://github.com/CoreySpohn/spaceodyssey/commit/e82832e642216965f83114535fbb2120ccf474da))
* **studies:** survey 3, priors with a declared pedigree and window-valued scheduling ([02e55ad](https://github.com/CoreySpohn/spaceodyssey/commit/02e55ad6783611959930e03ba81535be360c0268))
* **studies:** survey 4, a precursor exported from an upstream run and imported with its lineage ([de6f07d](https://github.com/CoreySpohn/spaceodyssey/commit/de6f07d988d8e34b5a2bf58f8b847429774d9a06))
* **studies:** survey 5, astrometric candidates on a delayed catalog roster ([30e6b01](https://github.com/CoreySpohn/spaceodyssey/commit/30e6b01898ec23a9daae8382cada960981535da8))
* **studies:** survey 6, a value that reads the age of the accepted prior record ([8a1495e](https://github.com/CoreySpohn/spaceodyssey/commit/8a1495e5886537551a9ec82584495ce74f15a16a))
* **studies:** verify_lineage checks a document's tables, their files and declared upstream hash (D004) ([ea88a27](https://github.com/CoreySpohn/spaceodyssey/commit/ea88a2724f23ad14412f1b2a5246f73834b8ae45))


### Bug Fixes

* **adapters:** cadence codes as the profile table states them ([a625907](https://github.com/CoreySpohn/spaceodyssey/commit/a6259078f1b66475fa51df9bcdcb48b276be4d2e))
* **adapters:** calibration id covers every budget declaration, per-law survey ids, Monte Carlo panels slow ([88a042c](https://github.com/CoreySpohn/spaceodyssey/commit/88a042ca777e8bb9a020086ca85567aa8d6de9a1))
* **adapters:** finite summary centroid for a zero-length interval ([0b3983d](https://github.com/CoreySpohn/spaceodyssey/commit/0b3983db9f1d8da3750739657bd7f691f328e655))
* **adapters:** no fit on columns an audience cannot read; mocked status tests; name target conventions ([0ef0ea9](https://github.com/CoreySpohn/spaceodyssey/commit/0ef0ea9ef3e8c0348804829d6c96b092fc303fe8))
* **adapters:** rate-bounded support margin for certification, nonzero-node anchors ([45bb1c2](https://github.com/CoreySpohn/spaceodyssey/commit/45bb1c2341954c51e5d34b3c1fdba2530af39343))
* **adapters:** rates through jaxedith's QE-aware intermediates and finite-frame variance when present ([e8d651e](https://github.com/CoreySpohn/spaceodyssey/commit/e8d651e5cb77576b5a96713e533714dce7086b98))
* **adapters:** zero summary SNR for a zero-length interval ([e0ef50d](https://github.com/CoreySpohn/spaceodyssey/commit/e0ef50dd7cbeac323b02a2e6128f029e7b757767))
* **codec:** an integer too long to write is refused at its pointer in parsed objects and when hashing ([f5d6571](https://github.com/CoreySpohn/spaceodyssey/commit/f5d657155a6f0e49dc5f4530037ed2c915a12779))
* **codec:** array files are read and written below the trusted directory's real path without following any link ([d7b7d9e](https://github.com/CoreySpohn/spaceodyssey/commit/d7b7d9e6f593f73f53dbc3f822cb50b9c51e06b1))
* **codec:** check restored fields against converters, array annotations and __post_init__ ([c345340](https://github.com/CoreySpohn/spaceodyssey/commit/c3453400b0c6b3f820ecd69201ad528455e1ff15))
* **codec:** compare tuple defaults element-wise; test(core): idle-facility guard for key version 2 ([17976f4](https://github.com/CoreySpohn/spaceodyssey/commit/17976f4539df596208277f0bfe502cc3d4de0183))
* **codec:** document objects are checked for integers too long to write, and a shape no array can hold is refused at its pointer ([7ab9f8f](https://github.com/CoreySpohn/spaceodyssey/commit/7ab9f8f4a26aa5ff839fa706921d2c49e21c955e))
* **codec:** keep dict insertion order in documents and their hash ([381b2a9](https://github.com/CoreySpohn/spaceodyssey/commit/381b2a90bf786fd8565aaa5cd9d5231f99847fe3))
* **codec:** locate pre-flight findings at their values and report independent findings together ([0f084e4](https://github.com/CoreySpohn/spaceodyssey/commit/0f084e4338c961f6cee63895721e4e538e1c2b97))
* **codec:** open the files directory one component at a time, refusing a symlinked directory or parent ([13ec49f](https://github.com/CoreySpohn/spaceodyssey/commit/13ec49f9497de640db4e5062f525f6a9238810e8))
* **codec:** parser limits, nesting depth and a wrong envelope or schema are findings in the lineage and pre-flight paths ([0b4c463](https://github.com/CoreySpohn/spaceodyssey/commit/0b4c463fba096ccbc23a3aab224db3918be25e7c))
* **codec:** rebuild the refit boundary only with the run document's registry; a study document with a run record gets a D003 warning ([b01b33a](https://github.com/CoreySpohn/spaceodyssey/commit/b01b33a53a70f9e79d622f443558a7a72b1b47f6))
* **codec:** refuse arrays in a non-native byte order ([932db5b](https://github.com/CoreySpohn/spaceodyssey/commit/932db5b2a7ecf36c12e2e68c731b53e2896db091))
* **codec:** refuse overflowing numbers, malformed JSON and repeated keys at their pointers ([15ad9e6](https://github.com/CoreySpohn/spaceodyssey/commit/15ad9e6651e156b4d4d28a083854c741f431e39d))
* **codec:** require a positive integer version default for every registered kind ([fbb25da](https://github.com/CoreySpohn/spaceodyssey/commit/fbb25da2765091f306909ef8ab413b7c362a1140))
* **codec:** require an enum under $enum and a module or dataclass under $kind ([1e9caab](https://github.com/CoreySpohn/spaceodyssey/commit/1e9caab85df779beb2bb6969492bc0e71ded0298))
* **codec:** require safe-open support, recheck boundary hash, guard envelopes, refuse growing files ([7db2827](https://github.com/CoreySpohn/spaceodyssey/commit/7db282736cd8924813e059af011ee95544012e98))
* **codec:** safe array file reads and writes, pre-flight shares run validation with load_run, files forwarded ([a8cc20a](https://github.com/CoreySpohn/spaceodyssey/commit/a8cc20a68d285f8d946d5f4a3cbc8a7c8a5b30d5))
* **codec:** the writer makes missing directories through the descriptor walk, never through a path ([271e69e](https://github.com/CoreySpohn/spaceodyssey/commit/271e69e9603716d7d54b383ecdf72f431a2a3feb))
* **core:** epoch comparators know tick columns by name, not by sign ([8eb98eb](https://github.com/CoreySpohn/spaceodyssey/commit/8eb98eb34e2a3a1e37ff184ede2f0e4213824c9d))
* **core:** grid position checks room before the ordinal increment ([2146c76](https://github.com/CoreySpohn/spaceodyssey/commit/2146c762e96444463b7820ca13ff29069b0ba57a))
* **core:** hold install needs completion budget, per-audience supersession, reserved capacity, padding-free digest, real dedup, safe grid ticks ([9e7b63b](https://github.com/CoreySpohn/spaceodyssey/commit/9e7b63b2827329954f4a677dcb65590d7a6a5bbe))
* **core:** key_version is a locatable study slot for pre-flight findings ([624f25e](https://github.com/CoreySpohn/spaceodyssey/commit/624f25e69f87d81afb90988ba645aaa4503722de))
* **core:** kit challenges boolean leaves on install and calibrates refits at every row depth ([96fa4ea](https://github.com/CoreySpohn/spaceodyssey/commit/96fa4ea0a62de250c037a71c73a33e9d995048f8))
* **core:** kit declares stochastic and subject-owned leaves, replays presented records and calibrates refits ([a0a411a](https://github.com/CoreySpohn/spaceodyssey/commit/a0a411a676c324120a872e3272904ba5c48a1b01))
* **core:** kit judges margins per stratum with t intervals, challenges install, gates refit_rows order and bands calibration by world ([de6caac](https://github.com/CoreySpohn/spaceodyssey/commit/de6caac1bda55c309d69da43e395dee3152b2545))
* **core:** kit reads log columns by name and reports status codes, clearing the dispatch lints ([6553817](https://github.com/CoreySpohn/spaceodyssey/commit/6553817e3805538cb189c8bdf7d590f0dc0a8e25))
* **core:** kit text avoids the denied core words ([0d2fa9f](https://github.com/CoreySpohn/spaceodyssey/commit/0d2fa9f95f4241cd6dda24351296cb930093b2bb))
* **core:** kit validates declarations without numbers ABCs (dispatch scan) ([cc5904f](https://github.com/CoreySpohn/spaceodyssey/commit/cc5904f78fc08107354a9af38d5ff18efbaefaa1))
* **core:** name the id counter plainly in replay ([09b8753](https://github.com/CoreySpohn/spaceodyssey/commit/09b87537727dcbe588a231913f401aaf08c302b5))
* **core:** refit_rows reads the log without a dynamic attribute read ([b632a31](https://github.com/CoreySpohn/spaceodyssey/commit/b632a319ab81061d566fa242046bb47fa02faed6))
* **core:** refuse corrections that involve a derived kind; unique scripted pairs; pedigree, release and fingerprint tests ([2dc469a](https://github.com/CoreySpohn/spaceodyssey/commit/2dc469acbd8a5bc3025bf4b69be49517320f5c08))
* **core:** replay folds, ids, phase and refusals ([0696299](https://github.com/CoreySpohn/spaceodyssey/commit/06962997ce0f92cf8dfff2e645cbc74316119a9c))
* **core:** report refusals per checkpoint and fold replay_belief rows at fold time ([38cf4d4](https://github.com/CoreySpohn/spaceodyssey/commit/38cf4d4cd3ada7169ba45b5f26a502990175d6ac))
* **core:** requeue copies rows without dynamic getattr; release handler keeps its fold lines ([acd018b](https://github.com/CoreySpohn/spaceodyssey/commit/acd018b076b1f839c122971f26d38eea66eecd21))
* **core:** scripted ids from the shared allocator; derived rows skip replay, supersession and fingerprint gaps ([f31f661](https://github.com/CoreySpohn/spaceodyssey/commit/f31f66192373f9ce4daba063c14b832dafb26f61))
* **core:** the kit's prior checks name and refuse an observer without a table and weights ([d115177](https://github.com/CoreySpohn/spaceodyssey/commit/d11517736b06f530f30e5262d7a8f461aa3664c8))
* **core:** the zero-weight probe carries the import's metadata, prior checks refuse non-binary weights and tables with nothing to judge ([1bb9614](https://github.com/CoreySpohn/spaceodyssey/commit/1bb9614e4290de95c0521586ee46c74f8fba5e55))
* **core:** the zero-weight probe keeps the release stamps unset, as at fold time ([76dbe96](https://github.com/CoreySpohn/spaceodyssey/commit/76dbe96f35877987a484edec1382187c28093004))
* **plugins:** refuse one class registered under two entry-point ids ([93c2601](https://github.com/CoreySpohn/spaceodyssey/commit/93c2601ef885a8fdae36032bcc7e798a5f5a9093))
* **preflight:** carry a locator on core findings and close the document-validation gaps ([9134cac](https://github.com/CoreySpohn/spaceodyssey/commit/9134cacb9b23925904b4bbe7bb1c9be8dbc1bec9))
* **studies:** a local index is unique per channel and subject (D006); the prior observer's row lookup and the exporter's clash check include the subject ([158cc8e](https://github.com/CoreySpohn/spaceodyssey/commit/158cc8e0e09c9f909f369d459a78c6e3c56b3042))
* **studies:** a row an audience never receives is not dropped for it, NaN-safe prior sums, no boundary findings without checkpoints ([d590c69](https://github.com/CoreySpohn/spaceodyssey/commit/d590c696d8de2a98dcd429b5910b56f3b09b1a34))
* **studies:** a table's upstream_hash is a string; an array there is refused at its pointer ([7142eca](https://github.com/CoreySpohn/spaceodyssey/commit/7142ecab081eb4946c3f16369458cfe74fa3ac8e))
* **studies:** accept a retained table prior on the observer-init boundary when both audiences resolve to the same prior (D010 amendment) ([125c480](https://github.com/CoreySpohn/spaceodyssey/commit/125c480f72f338cc20282df38be1d8168f40a5fb))
* **studies:** empty imports skip the width check and relative hook locators keep their detail ([7fe3e24](https://github.com/CoreySpohn/spaceodyssey/commit/7fe3e24f1f81ddd3b7f29039c9d002ed32c8de42))
* **studies:** only the audience-prior boundary takes a retained table prior (A10); equal resolutions on the observer-init boundary are D010 ([d5f3893](https://github.com/CoreySpohn/spaceodyssey/commit/d5f38936d9838f5deec21697d29f0902e534b4db))
* **studies:** payload rows, empty imports, audience-aware battery matching, JSON budget and locator guards ([10d31aa](https://github.com/CoreySpohn/spaceodyssey/commit/10d31aa6ab0db2eee9dfd0846367bc3d6cad8c10))
* **studies:** per-subject calibration, complete effect comparison, adaptive OFTI policy, disjoint candidate ids and finer tiering ([0dc8842](https://github.com/CoreySpohn/spaceodyssey/commit/0dc8842f59c260c7740feeaf40a89bb85582fc1a))
* **studies:** readable stand-in scope in saved run documents and a medium executor-parity guard ([d3b24b0](https://github.com/CoreySpohn/spaceodyssey/commit/d3b24b0b3ba6668b3e3f2cabba3e4c87cb32461b))
* **studies:** the subject map selects exported subjects, so one subject of a shared channel can be exported ([b1e19a2](https://github.com/CoreySpohn/spaceodyssey/commit/b1e19a236c1941c31a167791f0ae27f31091f93a))


### Performance Improvements

* **core:** reuse compiled snapshot, cancellation and admission kernels ([178c572](https://github.com/CoreySpohn/spaceodyssey/commit/178c572d2324d0c178d219468ca4fbde327fdbe7))

## [0.1.0](https://github.com/CoreySpohn/spaceodyssey/compare/v0.0.1...v0.1.0) (2026-11-05)


### ⚠ BREAKING CHANGES

* **plan:** PlanSummary.rv_precision_ms is replaced by k_precision_ms, the precision of the K estimate, with an optional k_precision_assumed_ms; CONTRACT_VERSION is 3. Producers convert per-epoch RV precision to K precision.

### Features

* add spaceodyssey.core transaction core with typed study contracts and an x64 guard ([9934681](https://github.com/CoreySpohn/spaceodyssey/commit/9934681e564226bc235887e4d36b434d6edc84a5))
* **analytic:** replicate-based yield intervals ([7c477e3](https://github.com/CoreySpohn/spaceodyssey/commit/7c477e390c5d82dd6912428cd9c81ae5c3d4c844))
* **certify:** abstract profile certificate interface and profile-hash check ([f029286](https://github.com/CoreySpohn/spaceodyssey/commit/f029286c060898285bda47219b658d74f2ade6ce))
* **certify:** closed-form Gaussian profile certifier ([75d8816](https://github.com/CoreySpohn/spaceodyssey/commit/75d8816ac01358ffb3a92e724dfd110fd30d2763))
* **certify:** implicit-function gradient for the interval radius ([a0a6207](https://github.com/CoreySpohn/spaceodyssey/commit/a0a62077a87aabb39ada2ba044772dc056a9c080))
* **certify:** plan-summary and outcome contracts, toy Gaussian certifier, analytic integrate engine ([18ec6cd](https://github.com/CoreySpohn/spaceodyssey/commit/18ec6cdf5e6844360feb926a6b858300090a98f3))
* **core:** coalesce wakes per tick, admit groups whole, mask effects per phase, order coincident faults, reserve ledger fields ([bd134e6](https://github.com/CoreySpohn/spaceodyssey/commit/bd134e62f55444397049a4cfb0c86aae0a95d926))
* **core:** conformance runner and the invariants every corpus case must hold ([e6654a9](https://github.com/CoreySpohn/spaceodyssey/commit/e6654a93c1eb47f55f4d666fb291658eaa070130))
* **core:** generated battery of null, swap, latency, fork, metamorphic and declaration checks ([8bde3e4](https://github.com/CoreySpohn/spaceodyssey/commit/8bde3e4b2658f2dab0c92fab5d3e3b044da8082d))
* **core:** key policy draws by phase, decision tick and ordinal, add coupling and asset and branch coordinates ([cb43a92](https://github.com/CoreySpohn/spaceodyssey/commit/cb43a92c7ddc831c367cde188c649a7bb84aff19))
* **core:** let a study carry a fork and a coupling into measurement draws ([3d6d9ba](https://github.com/CoreySpohn/spaceodyssey/commit/3d6d9baf98385f1e4f15f9b438e555ba8d52c8c3))
* **core:** record kinds, column audiences, statuses and revisions ([f02e72c](https://github.com/CoreySpohn/spaceodyssey/commit/f02e72c94e9b068d20ff11a9209602babb238133))
* **core:** release versions, snapshot fingerprints and the host protocol ([12ce040](https://github.com/CoreySpohn/spaceodyssey/commit/12ce040eb18f1c461e44f8dc03314471de7b3d60))
* **core:** run status, named overflow capacities, run_worlds and log replay ([fb7ed3c](https://github.com/CoreySpohn/spaceodyssey/commit/fb7ed3cf85794e022d876277e3d70f093fa08cf1))
* **core:** structural pre-flight with a study graph and declarations ([cef87bc](https://github.com/CoreySpohn/spaceodyssey/commit/cef87bc2bb1fbf1d1360bf646efa4977a3b1643f))
* **gaussian:** deterministic fixed-node Genz rectangle probability ([5c7382a](https://github.com/CoreySpohn/spaceodyssey/commit/5c7382a7c36382abfe49d793179d01d22e66b785))
* **inputs:** typed input codec ([026863b](https://github.com/CoreySpohn/spaceodyssey/commit/026863b9d3440a41d4ae0f0535fa554b6bf7e191))
* **ledger:** add reserved debit and refund accounting counters ([c4e3acc](https://github.com/CoreySpohn/spaceodyssey/commit/c4e3acc24552cc02b9f40b62568b34c65298b382))
* lint the core for domain vocabulary and physics imports ([182aa9b](https://github.com/CoreySpohn/spaceodyssey/commit/182aa9b38284795a851324e7c06f61448443a580))
* migrate the campaign core from hwosim ([bb7a555](https://github.com/CoreySpohn/spaceodyssey/commit/bb7a555e8c6f0ff6da25e8588d63354f877ed7a1))
* **plan:** claim axis and profile hash on certification outcomes ([6f7e092](https://github.com/CoreySpohn/spaceodyssey/commit/6f7e092fd1ebb81b484eabd8e438eddbe34ff613))
* **plan:** PlanSummary v2 with wavelengths, physical noise and RV precision ([76ba7f9](https://github.com/CoreySpohn/spaceodyssey/commit/76ba7f96e8ed6ebb2317b290316a0fa12773e4ca))
* **plan:** PlanSummary version 3 with true and assumed K precision ([11dc02b](https://github.com/CoreySpohn/spaceodyssey/commit/11dc02b60e29abf4af957fd3d1527e896379b557))
* **profile:** content-hashed Profile with edge claims and reference profiles ([e3095a5](https://github.com/CoreySpohn/spaceodyssey/commit/e3095a5e31196c4212658d2e31ca00e5528bbdad))
* **profile:** evidence_rule none drops the prior-only evidence condition ([9415345](https://github.com/CoreySpohn/spaceodyssey/commit/94153456a3f3101e1331839d7b00d67aa6462776))
* **reporting:** move yield reporting into spaceodyssey.reporting and export certified_yield at the root ([d973745](https://github.com/CoreySpohn/spaceodyssey/commit/d973745b8ffa9cb9d49a32e068b02e3a1d0aa346))
* **studies:** add frozen network policy and survey conformance ([7f1fe33](https://github.com/CoreySpohn/spaceodyssey/commit/7f1fe3374611f360cbefc8c04ee65e7ed9de3a72))
* **studies:** move the corpus into the library and add six conformance cases ([f502676](https://github.com/CoreySpohn/spaceodyssey/commit/f502676b9c02c24eb2803ad09e651803c0e0a3de))
* **truth:** truth models, prior sets and observed-data contracts ([6787858](https://github.com/CoreySpohn/spaceodyssey/commit/678785886e37423acfa82d2f7c5514665a8c1ff6))
* **viz:** add plot_certification_curve ([8fa3e7a](https://github.com/CoreySpohn/spaceodyssey/commit/8fa3e7a5e095d9a4a87b6411cad822387e0284d3))
* **viz:** add plot_claim_posterior ([2684978](https://github.com/CoreySpohn/spaceodyssey/commit/2684978c13faf14af127c8b7f592c3e5c5565e5a))
* **viz:** add plot_plan_visits and plot_observed_data ([e99bfd5](https://github.com/CoreySpohn/spaceodyssey/commit/e99bfd5675906a62e0a0ef9c815284c4cbb838da))
* **viz:** add plot_planet_plane ([7145900](https://github.com/CoreySpohn/spaceodyssey/commit/71459003c22d972ec5d5acccfd7acfe9d915c735))
* **viz:** add plot_tier_agreement ([11aea6c](https://github.com/CoreySpohn/spaceodyssey/commit/11aea6c2cf2fab0aa194c154558cde375703de44))
* **viz:** add plot_yield_comparison ([cb2bde4](https://github.com/CoreySpohn/spaceodyssey/commit/cb2bde4700475f5a27ca8e6d73dda0d10261e60d))
* **viz:** add plot_yield_sweep ([eded3e3](https://github.com/CoreySpohn/spaceodyssey/commit/eded3e3603c6676e10d2aacdb1476e145b38b88b))
* **viz:** add viz package scaffold, lazy exports and pure extraction helpers ([c6580ac](https://github.com/CoreySpohn/spaceodyssey/commit/c6580ac1555cf38f131a990b2d64c05329dd10d2))


### Bug Fixes

* **analytic:** bound the certified-yield standard error by both channel errors ([10211ad](https://github.com/CoreySpohn/spaceodyssey/commit/10211ad37e7d981b9e01e6b1c8800e1151acb664))
* **analytic:** count only correct certifications in the certified yield ([bc207b0](https://github.com/CoreySpohn/spaceodyssey/commit/bc207b0344728a7ed331558027c0fb120a42523e))
* **api:** export the observe function, not the observe submodule ([2208591](https://github.com/CoreySpohn/spaceodyssey/commit/22085914d810009d76a2d1fe024073577f7c9b90))
* **battery:** find the view of a decision that wakes on the step ending the run ([44b3790](https://github.com/CoreySpohn/spaceodyssey/commit/44b37901ccaab576cbfcbfa54730782c292a6174))
* **certify:** finite gradients at infinite region bounds ([fddcd23](https://github.com/CoreySpohn/spaceodyssey/commit/fddcd23fc903ae2d65bf0d7d9610d50f043ba2aa))
* **certify:** NaN interval bounds propagate ([bd32153](https://github.com/CoreySpohn/spaceodyssey/commit/bd32153467fefc8f5892e9fc367b39906e1f6a4d))
* **certify:** review fixes for profile validation and the Gaussian profile certifier ([5b19db3](https://github.com/CoreySpohn/spaceodyssey/commit/5b19db3696e40cc0fe481714d162bd74966a0995))
* **certify:** safe gradients for empty intervals and context-scaled quantities ([978c62e](https://github.com/CoreySpohn/spaceodyssey/commit/978c62eb8d2c4f405cd3e2bc26ecb06fa585701b))
* **conformance:** judge exact decision views, admissions, folds and overflow balance ([eab596f](https://github.com/CoreySpohn/spaceodyssey/commit/eab596fefd51ed1cba408adf49b3d2c0335f5ea1))
* **core:** compare every carry and trace field in the canonical history ([2dfb4a6](https://github.com/CoreySpohn/spaceodyssey/commit/2dfb4a6628a17feae638fc5d21ce3a6317fce7e8))
* **core:** compare ledger debit and refund counters across executors and allow the parity field walk ([d5e5a0d](https://github.com/CoreySpohn/spaceodyssey/commit/d5e5a0d1a6bf3340f6654d4877da7ffd49e25495))
* **core:** count folds by channel, subject and admission index, and skip rows the observer cannot read in replay ([9aa5b47](https://github.com/CoreySpohn/spaceodyssey/commit/9aa5b476462fb0343c1dd5678a0268f3f988ef4f))
* **core:** enforce audience isolation and result ordering ([37534b5](https://github.com/CoreySpohn/spaceodyssey/commit/37534b59c8fc8c77a6ad453c2ba0ff5ecebbb46a))
* **core:** enforce battery and declaration checks ([d204f09](https://github.com/CoreySpohn/spaceodyssey/commit/d204f0996533910257c634390cc525c228d700b6))
* **core:** enforce conformance accounting and record checks ([8aceeed](https://github.com/CoreySpohn/spaceodyssey/commit/8aceeed6f3b3c36b4b5e49f4605751e203106b14))
* **core:** leave snapshot and event digests out of the relabel, epoch and latency comparisons ([611bcd4](https://github.com/CoreySpohn/spaceodyssey/commit/611bcd48628310d03a61d84954eb22efb8714e80))
* **core:** match relabel fields by leaf name and map the record source id ([ae95d60](https://github.com/CoreySpohn/spaceodyssey/commit/ae95d60c991771efe770c1be28f0d34d47963524))
* **core:** preserve executor coupling and parity contracts ([872ffac](https://github.com/CoreySpohn/spaceodyssey/commit/872ffac8d16b2d64e58ff6a924f55e18479cfb1d))
* **core:** refuse a study whose capacities carry no registered record kinds ([7395446](https://github.com/CoreySpohn/spaceodyssey/commit/73954462f5e4736bd0ea7792eddd8470226a82ee))
* **core:** reject complex leaves in tree_equal_report ([dafcab7](https://github.com/CoreySpohn/spaceodyssey/commit/dafcab7a43f6c86590b1b3e615041a231f3cff99))
* **core:** replay releases by exact log identity ([2a62e6f](https://github.com/CoreySpohn/spaceodyssey/commit/2a62e6fc4defec165f7d032ac88d77aae89e6950))
* **core:** scope each dispatch allowance to its rules, function and count and scan neighbouring spellings ([88b43db](https://github.com/CoreySpohn/spaceodyssey/commit/88b43db1247ba3f26b8598368aa3a8b2e676cf80))
* **core:** tag optional key coordinates so asset and branch never alias ([2e8581a](https://github.com/CoreySpohn/spaceodyssey/commit/2e8581a32ebbb1a22e973139f34dc2ef3b89f0d5))
* **core:** treat Query as a supported effect in pre-flight now that the host answers it ([46ae17c](https://github.com/CoreySpohn/spaceodyssey/commit/46ae17c5fef3e7570feb9117537b6bd6cd6b6485))
* **core:** widen the contract annotation walk to typevars, aliases and bare containers ([efc9c48](https://github.com/CoreySpohn/spaceodyssey/commit/efc9c485606d31201f6b3abb77191e04d6b30922))
* **deps:** require eyepiece&gt;=0.10.0 for ep.role ([c594109](https://github.com/CoreySpohn/spaceodyssey/commit/c5941091e0b0dbfcdd908b320d149c383d2050f7))
* **gaussian:** honor evidence_rule none in the Gaussian certifier ([60364fe](https://github.com/CoreySpohn/spaceodyssey/commit/60364fec7ab97f94e330158a28cfa3141f01e6c2))
* **inputs:** harden scalar, cycle, snapshot and plan field handling ([3c872a0](https://github.com/CoreySpohn/spaceodyssey/commit/3c872a0d1b94a9adfc764ad10daf5e828668d4ca))
* **inputs:** require plan arrays, refuse colliding keys, validate codec type ([094add9](https://github.com/CoreySpohn/spaceodyssey/commit/094add9997a6e76226ed5734bfb52acc8c7909ae))
* **io:** refuse to overwrite an existing run directory ([adb1d47](https://github.com/CoreySpohn/spaceodyssey/commit/adb1d477aaad0bf79796a644d2de036225c923f2))
* **lint:** reject every reference to dynamic loaders and name-looked-up callees in the core import lint ([94f5e72](https://github.com/CoreySpohn/spaceodyssey/commit/94f5e7217d8b6b969cb368a8bacf5d078c13e6f0))
* **plan:** reject complex K precisions ([9815c5a](https://github.com/CoreySpohn/spaceodyssey/commit/9815c5a4c6303e7c44ab089deed7b6328160a5c9))
* **profile:** construction-order guard, predictive validation, canonical hashing and neutral reference id ([bb13a1f](https://github.com/CoreySpohn/spaceodyssey/commit/bb13a1f67f1d1f29c1f77c572b88790cd9bc6e5d))
* **registry:** list every built-in adapter module for lazy registration ([8071104](https://github.com/CoreySpohn/spaceodyssey/commit/8071104de8b83ee2f9c4ae0bf63e1374a144dc50))
* **spaceodyssey:** accept open and closed jaxprs in the affine check ([01a5653](https://github.com/CoreySpohn/spaceodyssey/commit/01a5653f527356dfe9e5e40a20d2a3f1ab6b5d8c))
* **spaceodyssey:** match host executor run keyword arguments ([b78dc6c](https://github.com/CoreySpohn/spaceodyssey/commit/b78dc6c5de104165f08c24179342cbaca61d6c45))
* **spaceodyssey:** raise WiringError for a missing certification choice ([40e7891](https://github.com/CoreySpohn/spaceodyssey/commit/40e7891ffb6de39dcf2936bc6ff02dec8f3b2d50))
* **spaceodyssey:** read x64 through public JAX config API ([21a0ed7](https://github.com/CoreySpohn/spaceodyssey/commit/21a0ed74d693785579a32cff025727d1302d5799))
* **spaceodyssey:** reject missing certification choice ([1398d64](https://github.com/CoreySpohn/spaceodyssey/commit/1398d64374cf6188e758902b623f8c39dc0bfca3))
* **spaceodyssey:** reject missing mission loop seams ([d2f42db](https://github.com/CoreySpohn/spaceodyssey/commit/d2f42dbc2b1bcb731eb2c0680e823186d8c2d1f9))
* **spaceodyssey:** reject missing truth scene ([c0241c6](https://github.com/CoreySpohn/spaceodyssey/commit/c0241c68b04e6ce37a0eb1bedc56a90032a22ded))
* **spaceodyssey:** respect x64 when tracing quantities ([0d51f34](https://github.com/CoreySpohn/spaceodyssey/commit/0d51f34627e0de037ec5921a53981ec981a078f3))
* **studies:** require a static network observation limit, pin survey 1 timing and correct conformance docs ([8b5ecb6](https://github.com/CoreySpohn/spaceodyssey/commit/8b5ecb6d8e2959f128e8633b364e6008ab8d9792))
* **testing:** check integrate-only certificates without calling initial ([ed683f4](https://github.com/CoreySpohn/spaceodyssey/commit/ed683f4cfad247e88abe831f7bf228526e07e508))
* **viz:** legible log ticks, claim titles, marker keys and legend placement ([e7fe07f](https://github.com/CoreySpohn/spaceodyssey/commit/e7fe07fdb5202d243d006272dd14490bd921c397))
* **viz:** wrap the yield sweep y label so it stays inside the figure ([d7ad96c](https://github.com/CoreySpohn/spaceodyssey/commit/d7ad96cc1d1b703c1d86c666425d7e4ceceb4056))
* **wiring:** validate the same registry entry that build constructs ([1e5c752](https://github.com/CoreySpohn/spaceodyssey/commit/1e5c752646477c53f90555360336ac2e76ebde6d))


### Performance Improvements

* **core:** record decision views through a slot so equal studies share one compile ([bbcb76b](https://github.com/CoreySpohn/spaceodyssey/commit/bbcb76b41739fc7bf21fcff0b8056b240c2c8ce8))
* **core:** reuse the snapshot digest scan body ([4a4754a](https://github.com/CoreySpohn/spaceodyssey/commit/4a4754aec8d441d9a26e73dad1e336f62b77b747))
* **studies:** hold plan, wait, stop and horizon knobs as data so studies share compiled programs ([12b1227](https://github.com/CoreySpohn/spaceodyssey/commit/12b122780f7f3e788b7b318f6dfd659ec6eb50b5))

## 0.0.1 (2026-09-16)


### Features

* scaffold the spaceodyssey package ([b746a98](https://github.com/CoreySpohn/spaceodyssey/commit/b746a982768ca2f8cf020f5d3438e27423984d9f))


### Miscellaneous Chores

* release 0.0.1 ([a3e18b0](https://github.com/CoreySpohn/spaceodyssey/commit/a3e18b047598118ba2cd01053aa68c28b673445d))
