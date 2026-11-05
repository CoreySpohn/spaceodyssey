# Changelog

## Unreleased

### Features

* add a JAX/Equinox transaction core for discrete-event observing campaigns, with shared event ordering, atomic admission, resource accounting and semantic random keys; host and compiled executors; typed records with audiences, release times and revisions; a snapshot-bound host request/result protocol; structural pre-flight checks; a generated validation battery; and a conformance corpus with hand histories. Run the acceptance gate with `pytest tests/test_slice1_gate.py`.

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
