# IsoLens

<p align="center">
  <img src="IsoLens_Logo.png" alt="IsoLens" width="300">
</p>

**IsoLens** is a Bayesian framework for inferring microlensing lens properties by combining disk/bulge age–metallicity prior distributions with multi-isochrone stellar models.

<!-- mesh independence (i.e., different sampling sizes lead to same mean and uncertainty) should be verified by user, due to different sizes of errorbar (Lens Mag, murel_hel) -->

<!-- the resolution of the logo should be lower, current 1.3MB too large -->

## Motivation

For microlensing lenses with measurable flux, the conversion between lens luminosity and mass is not unique because the mass–luminosity relation depends on stellar age and metallicity, particularly for higher-mass lenses. IsoLens addresses this issue by combining multi-isochrone stellar models with disk and bulge age–metallicity priors, allowing these population uncertainties to be properly marginalized when inferring lens properties.

## Citation

If you use IsoLens in your research, please cite:

```bibtex
@software{Zhang_IsoLens_2026,
  author = {Zhang, Jiyuan},
  month = aug,
  title = {{IsoLens}},
  url = {https://github.com/zhangjiyuan22/IsoLens},
  year = {2026}
}
