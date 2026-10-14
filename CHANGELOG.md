# Changelog

## [1.0.0](https://github.com/CoreySpohn/spaceodyssey/compare/v0.0.1...v1.0.0) (2026-10-14)


### ⚠ BREAKING CHANGES

* **plan:** PlanSummary.rv_precision_ms is replaced by k_precision_ms, the precision of the K estimate, with an optional k_precision_assumed_ms; CONTRACT_VERSION is 3. Producers convert per-epoch RV precision to K precision.

### Features

* **analytic:** replicate-based yield intervals ([7c477e3](https://github.com/CoreySpohn/spaceodyssey/commit/7c477e390c5d82dd6912428cd9c81ae5c3d4c844))
* **certify:** abstract profile certificate interface and profile-hash check ([f029286](https://github.com/CoreySpohn/spaceodyssey/commit/f029286c060898285bda47219b658d74f2ade6ce))
* **certify:** closed-form Gaussian profile certifier ([75d8816](https://github.com/CoreySpohn/spaceodyssey/commit/75d8816ac01358ffb3a92e724dfd110fd30d2763))
* **certify:** implicit-function gradient for the interval radius ([a0a6207](https://github.com/CoreySpohn/spaceodyssey/commit/a0a62077a87aabb39ada2ba044772dc056a9c080))
* **certify:** plan-summary and outcome contracts, toy Gaussian certifier, analytic integrate engine ([18ec6cd](https://github.com/CoreySpohn/spaceodyssey/commit/18ec6cdf5e6844360feb926a6b858300090a98f3))
* **gaussian:** deterministic fixed-node Genz rectangle probability ([5c7382a](https://github.com/CoreySpohn/spaceodyssey/commit/5c7382a7c36382abfe49d793179d01d22e66b785))
* **inputs:** typed input codec ([026863b](https://github.com/CoreySpohn/spaceodyssey/commit/026863b9d3440a41d4ae0f0535fa554b6bf7e191))
* migrate the campaign core from hwosim ([bb7a555](https://github.com/CoreySpohn/spaceodyssey/commit/bb7a555e8c6f0ff6da25e8588d63354f877ed7a1))
* **plan:** claim axis and profile hash on certification outcomes ([6f7e092](https://github.com/CoreySpohn/spaceodyssey/commit/6f7e092fd1ebb81b484eabd8e438eddbe34ff613))
* **plan:** PlanSummary v2 with wavelengths, physical noise and RV precision ([76ba7f9](https://github.com/CoreySpohn/spaceodyssey/commit/76ba7f96e8ed6ebb2317b290316a0fa12773e4ca))
* **plan:** PlanSummary version 3 with true and assumed K precision ([11dc02b](https://github.com/CoreySpohn/spaceodyssey/commit/11dc02b60e29abf4af957fd3d1527e896379b557))
* **profile:** content-hashed Profile with edge claims and reference profiles ([e3095a5](https://github.com/CoreySpohn/spaceodyssey/commit/e3095a5e31196c4212658d2e31ca00e5528bbdad))
* **profile:** evidence_rule none drops the prior-only evidence condition ([9415345](https://github.com/CoreySpohn/spaceodyssey/commit/94153456a3f3101e1331839d7b00d67aa6462776))
* **reporting:** move yield reporting into spaceodyssey.reporting and export certified_yield at the root ([d973745](https://github.com/CoreySpohn/spaceodyssey/commit/d973745b8ffa9cb9d49a32e068b02e3a1d0aa346))
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
* **certify:** finite gradients at infinite region bounds ([fddcd23](https://github.com/CoreySpohn/spaceodyssey/commit/fddcd23fc903ae2d65bf0d7d9610d50f043ba2aa))
* **certify:** NaN interval bounds propagate ([bd32153](https://github.com/CoreySpohn/spaceodyssey/commit/bd32153467fefc8f5892e9fc367b39906e1f6a4d))
* **certify:** review fixes for profile validation and the Gaussian profile certifier ([5b19db3](https://github.com/CoreySpohn/spaceodyssey/commit/5b19db3696e40cc0fe481714d162bd74966a0995))
* **certify:** safe gradients for empty intervals and context-scaled quantities ([978c62e](https://github.com/CoreySpohn/spaceodyssey/commit/978c62eb8d2c4f405cd3e2bc26ecb06fa585701b))
* **deps:** require eyepiece&gt;=0.10.0 for ep.role ([c594109](https://github.com/CoreySpohn/spaceodyssey/commit/c5941091e0b0dbfcdd908b320d149c383d2050f7))
* **gaussian:** honor evidence_rule none in the Gaussian certifier ([60364fe](https://github.com/CoreySpohn/spaceodyssey/commit/60364fec7ab97f94e330158a28cfa3141f01e6c2))
* **inputs:** harden scalar, cycle, snapshot and plan field handling ([3c872a0](https://github.com/CoreySpohn/spaceodyssey/commit/3c872a0d1b94a9adfc764ad10daf5e828668d4ca))
* **inputs:** require plan arrays, refuse colliding keys, validate codec type ([094add9](https://github.com/CoreySpohn/spaceodyssey/commit/094add9997a6e76226ed5734bfb52acc8c7909ae))
* **io:** refuse to overwrite an existing run directory ([adb1d47](https://github.com/CoreySpohn/spaceodyssey/commit/adb1d477aaad0bf79796a644d2de036225c923f2))
* **plan:** reject complex K precisions ([9815c5a](https://github.com/CoreySpohn/spaceodyssey/commit/9815c5a4c6303e7c44ab089deed7b6328160a5c9))
* **profile:** construction-order guard, predictive validation, canonical hashing and neutral reference id ([bb13a1f](https://github.com/CoreySpohn/spaceodyssey/commit/bb13a1f67f1d1f29c1f77c572b88790cd9bc6e5d))
* **registry:** list every built-in adapter module for lazy registration ([8071104](https://github.com/CoreySpohn/spaceodyssey/commit/8071104de8b83ee2f9c4ae0bf63e1374a144dc50))
* **testing:** check integrate-only certificates without calling initial ([ed683f4](https://github.com/CoreySpohn/spaceodyssey/commit/ed683f4cfad247e88abe831f7bf228526e07e508))
* **viz:** legible log ticks, claim titles, marker keys and legend placement ([e7fe07f](https://github.com/CoreySpohn/spaceodyssey/commit/e7fe07fdb5202d243d006272dd14490bd921c397))
* **viz:** wrap the yield sweep y label so it stays inside the figure ([d7ad96c](https://github.com/CoreySpohn/spaceodyssey/commit/d7ad96cc1d1b703c1d86c666425d7e4ceceb4056))
* **wiring:** validate the same registry entry that build constructs ([1e5c752](https://github.com/CoreySpohn/spaceodyssey/commit/1e5c752646477c53f90555360336ac2e76ebde6d))

## 0.0.1 (2026-09-16)


### Features

* scaffold the spaceodyssey package ([b746a98](https://github.com/CoreySpohn/spaceodyssey/commit/b746a982768ca2f8cf020f5d3438e27423984d9f))


### Miscellaneous Chores

* release 0.0.1 ([a3e18b0](https://github.com/CoreySpohn/spaceodyssey/commit/a3e18b047598118ba2cd01053aa68c28b673445d))
