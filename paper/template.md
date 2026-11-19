---
title: "{{title}}"
author: "David Oloyede"
date: "{{date}}"
---

# Abstract

We map land cover in 2025 over a {{box_km2}} km² box around the Mau Forest Complex in western Kenya from a cloud-masked median of {{n_scenes}} Sentinel-2 scenes taken from {{first_date_long}} to {{last_date_long}}, with Copernicus elevation and slope. Training labels were not drawn by hand: they are cells where ESA WorldCover 2021 and the Esri land cover map for 2023 agree on a class. A random forest separates forest, cropland and grassland, built-up land, and water and wetland. A separate plantation forest class, labelled from a 2015 forest management map, did not pass a rule written into the code before cross-validation was run (the rule was not pre-registered): its recall under 10 km blocked cross-validation was {{plant_rec}}, against a required {{threshold}}. Plantations are therefore mapped with forest, and no bare-ground class could be trained. Overall agreement with the map-agreement labels is {{f_rf_random}} with random cross-validation and {{f_rf_block10}} with 10 km spatial blocks; with the plantation class kept, it falls from {{c_rf_random}} to {{c_rf_block10}}. These figures measure agreement with labels from existing maps on uniform cells, not map accuracy. The map shows {{forest_km2}} km² of forest ({{forest_pct}}% of the box), against {{wc_forest_km2}} km² in WorldCover and {{es_forest_km2}} km² in the Esri map; it agrees with WorldCover on {{agree_wc_pct}}% of the box and with the Esri map on {{agree_es_pct}}%. Because the training labels are the cells where these two maps agree, this agreement is partly built in and is not evidence of accuracy. {{abstract_accuracy}}

**Keywords:** land cover; Sentinel-2; random forest; spatial cross-validation; area estimation; Mau Forest; Kenya

# 1. Introduction

The Mau Forest Complex in western Kenya is a group of montane forest blocks surrounded by smallholder farms, tea estates, tree plantations and towns. In the Eastern Mau, more than 40% of the forest land present in 1976 had been converted to other uses by 2014, and the conversion has been tied to land policy and political competition (Kweyu et al., 2020). Global land-cover maps now cover the area every year at 10 m, but they disagree with each other where forest, farms and planted trees mix, and the accuracy figures reported for them are assessed over large regions, not for a single landscape.

Two practical problems sit behind any new map of an area like this. The first is labels. Without field data, a map needs reference labels from somewhere, and labels drawn by one person from imagery are slow to collect and hard to check. Taking training labels from existing maps, filtered to keep reliable cells, is one recognised way to get many training labels quickly (Radoux et al., 2014; Zhang et al., 2021); here we use the cells where two maps agree. It carries the errors and the definitions of those maps into the new one. The second is how accuracy is tested. Random cross-validation puts neighbouring cells, which look alike, into both the training and the test folds, and can make a classifier look better than it is in places it has not seen; blocking the folds in space is one remedy (Roberts et al., 2017; Ploton et al., 2020).

This study asks three questions for 2025: which land-cover classes can be separated in the Mau Forest area from a dry-season Sentinel-2 composite with labels taken from existing maps; how much the cross-validated estimate changes when the folds are blocked in space; and how the new map compares with ESA WorldCover, the Esri land cover map and Hansen forest loss. Unbiased accuracy and area estimates need an independently collected probability sample of reference labels (Olofsson et al., 2014; Stehman & Foody, 2019). We looked for one and did not find enough points inside the box, so the study also provides a sample design and a labelling tool, and reports those estimates as pending.

# 2. Data and methods

## 2.1 Study area

The study area is a rectangle from {{lon0}} to {{lon1}}°E and {{lat0}} to {{lat1}}°S ({{box_km2}} km²), around the Mau Forest Complex in Narok, Nakuru, Bomet and Kericho counties. It is a box drawn around the forest blocks, not the legal forest boundary, and it includes farmland, tea estates, towns and part of Lake Nakuru. All rasters are on one grid of 20 m cells in WGS 84 / UTM zone 36S.

## 2.2 Sentinel-2 composite and terrain

We used Sentinel-2 Level-2A surface reflectance (Drusch et al., 2012) from the Microsoft Planetary Computer: all {{n_scenes}} scenes over tiles 36MYD, 36MYE, 36MZD and 36MZE from {{first_date_long}} to {{last_date_long}} with less than {{max_cloud}}% cloud cover. Scenes from one date and orbit were joined into one acquisition ({{n_acq}} acquisitions). Cells were kept where the scene classification layer marks vegetation, bare soil, water or unclassified, and the median of the kept values was taken for ten bands (B02 to B08, B8A, B11, B12; the 10 m bands averaged to 20 m). The median cell in the box has {{clear_median}} clear observations and 95% of cells have at least {{clear_p05}}. Four indices were added: NDVI, NDWI (green and near infrared), NDMI (near infrared and shortwave infrared 1) and an NBR-type index (near infrared and shortwave infrared 2). Elevation from the Copernicus DEM GLO-30 (ESA, 2022), resampled bilinearly to 20 m, and slope computed from it bring the total to {{n_features}} features per cell.

## 2.3 Classes and training labels

The candidate classes were forest, plantation forest, cropland and grassland, built-up, water and wetland, and bare ground. Training labels come from existing maps, not from field visits or image interpretation. A cell is labelled where ESA WorldCover 2021 v200 (Zanaga et al., 2022), aggregated to 20 m by the most common value, and the Esri / Impact Observatory annual land cover map for 2023 (Karra et al., 2021) put it in the same class, and where its whole 3 × 3 neighbourhood carries the same label, which keeps mixed edge cells out. The two maps agree, before the neighbourhood rule, on {{share_agree_pct}}% of the box. Neither map has a plantation class, so tree cells were split with the global forest management map for 2015 (Lesiv et al., 2022): plantation forest where it shows planted or plantation forest, forest where it shows naturally regenerating forest. Tree cells it calls agroforestry, which covers {{fml_agro_pct}}% of the box, received no tree label. Tree cells with Hansen loss in 2021–2024 were dropped, because the maps may predate the loss. From the eligible cells we drew a stratified random sample of up to {{n_per_class}} cells per class ({{n_train}} cells in all).

The class scheme was then set from what the data separate, by a rule written into the code before cross-validation was run (not pre-registered): plantation forest stays a class only if its precision and recall both reach {{threshold}} under 10 km blocked cross-validation with the random forest; otherwise its cells become forest.

## 2.4 Classifiers and cross-validation

The map is made with a random forest (Breiman, 2001) of 300 trees, at least two samples per leaf and the square root of the number of features tried at each split. A histogram gradient-boosting classifier (Friedman, 2001; scikit-learn, Pedregosa et al., 2011) with 300 iterations is the comparison. Neither was tuned. Both were evaluated with 5-fold cross-validation in two ways: random folds stratified by class, and spatial blocks of 5, 10 and 20 km on the UTM grid, assigned to folds so that classes are spread across folds (the SpatialBlockKFold class of the author's spatial-stats-core package, after Roberts et al., 2017). The out-of-fold predictions of all five folds are pooled before computing precision, recall, F1 and overall agreement. The blocks have no buffer between training and test cells, so cells on either side of a block edge can be close: the smallest distance from a test cell to a training cell is {{dist_min_10}} m for 10 km blocks, and the median distance, averaged over folds, is {{dist_med_10}} m, against {{dist_med_random}} m for random folds.

## 2.5 Comparison maps

WorldCover 2021 and Esri 2023 were translated into the study's classes (WorldCover tree cover to forest; shrubland, grassland and cropland to cropland and grassland; built-up; water, herbaceous wetland and mangroves to water and wetland; Esri trees, crops and rangeland, built area, water and flooded vegetation in the same way; bare ground to bare or other) and compared cell by cell with the 2025 map. Hansen Global Forest Change v1.12 (Hansen et al., 2013) gives tree cover in 2000 and the year of stand-replacing loss from 2001 to 2024. It is a separate product with its own definition of trees, and loss includes plantation harvest, so it is used as context, not as a check. Loss areas by year are computed on the Hansen grid inside the box.

## 2.6 Reference data and the planned accuracy assessment

We searched for open, independently collected reference points inside the box. The Geo-Wiki crowdsourced data of 2011–2012 (Fritz et al., 2016, 2017) have {{fritz_locations}} locations inside it. The Geo-Wiki global reference data set for 2015 at 10 m (Lesiv et al., 2025a, 2025b) has {{lesiv_sites}} sites, each a 100 m square labelled as 100 cells of 10 m. Neither set reaches the {{target_points}} points we aimed for, both predate 2025, and neither was drawn from the strata of this map, so they cannot give unbiased accuracy or area estimates. We use the {{lesiv_sites}} sites only for a descriptive check.

Instead, we drew a stratified random sample from the 2025 map with the map classes as strata. The total size follows Olofsson et al. (2014) and Cochran (1977): n = (Σ W~i~ S~i~ / S(Ô))², with W~i~ the mapped share of class i, S~i~ = √(U~i~(1 − U~i~)) for an assumed user's accuracy U~i~ ({{ua_assumed}}), and a target standard error of overall accuracy S(Ô) of {{target_se}}. This gives {{n_formula}} points ({{n_formula_01}} for a target of 0.01). Each class gets its proportional share, raised to at least {{min_per_class}} points for the small classes, which makes {{n_sample}} points: {{allocation}}. With {{n_built_sample}} points and an assumed user's accuracy of {{ua_built}}, the 95% interval on built-up user's accuracy would be about ±{{ua_built_hw}}. The points are labelled on high-resolution imagery with the labelling page that comes with the study, following a written protocol whose class definitions match the translation of WorldCover and Esri classes in Section 2.5. The labelling page shows only a location, not the map class; the interpreter is the map author and has seen the map, so labelling is blind per point, not fully independent. The labels then give the error matrix, overall, user's and producer's accuracy and stratified area estimates with 95% intervals (Olofsson et al., 2014).

# 3. Results

## 3.1 Composite and training sample

![**Figure 1.** Sentinel-2 median composite for January to March 2025, true colour. The forest blocks of the Mau Forest Complex are the dark green areas; Lake Nakuru is at the right edge.](paper/figures/fig1_composite.jpg)

The composite (Figure 1) has no gaps worth noting: {{clear_zero_pct}}% of cells in the box have no clear observation. Eligible training cells were plentiful for forest ({{elig_forest}}), cropland and grassland ({{elig_crop}}), built-up land ({{elig_built}}) and water and wetland ({{elig_water}}), but plantation forest had only {{elig_plant}} cells and bare ground had none after the neighbourhood rule ({{agree_bare}} agreeing cells before it), so no bare class was trained. Figure 2 shows where the training cells and the reference points lie.

![**Figure 2.** Where the training cells (left) and the reference sample (right) lie. Training cells are where ESA WorldCover 2021 and Esri 2023 agree in a uniform neighbourhood, before plantation was merged into forest; reference points are shown without their map class (the oversampled small strata still show as clusters).](paper/figures/fig_samples.jpg)

## 3.2 Class scheme and cross-validation

With the plantation class kept, both classifiers agree well with the labels under random cross-validation, but blocked cross-validation shows that plantations are mostly confused with forest in places the model has not seen (Table 1, Figure 3). Plantation recall falls from {{c_plant_rec_random}} with random folds to {{plant_rec}} with 10 km blocks, and forest precision from {{c_forest_prec_random}} to {{c_forest_prec_block10}}, because plantation cells are predicted as forest. The plantation class therefore failed the rule and was merged with forest. With the available plantation labels ({{plant_share_pct}}% of the box in the 2015 100 m map), plantations could not be separated from natural forest.

Table 1. Cross-validated agreement with map-agreement labels, random forest, with the plantation class kept (candidate scheme) and in the final scheme. Values pool the out-of-fold predictions of five folds.

{{cv_table}}

In the final scheme, agreement changes little between random and blocked folds for the random forest ({{f_rf_random}} random, {{f_rf_block5}}, {{f_rf_block10}} and {{f_rf_block20}} for 5, 10 and 20 km blocks) or for gradient boosting ({{f_gb_random}} random, {{f_gb_block10}} with 10 km blocks). The two models are within {{f_model_diff}} of each other under every fold scheme in the final class scheme. In the final scheme the forest class also holds the plantation cells, so it has twice as many training cells as each other class. Per class, under 10 km blocks, the random forest has precision and recall of {{f_forest_p}} and {{f_forest_r}} for forest, {{f_crop_p}} and {{f_crop_r}} for cropland and grassland, {{f_built_p}} and {{f_built_r}} for built-up land and {{f_water_p}} and {{f_water_r}} for water and wetland. Single folds under 20 km blocks range from {{f_fold20_min}} to {{f_fold20_max}}. The test folds keep the class balance of the training sample, not of the landscape, so precision for rare classes such as built-up land is overstated: reweighted to the mapped class shares, built-up precision under 10 km blocks would be {{built_prec_rw}} instead of {{f_built_p}}. The training cells are the cells on which two existing maps already agree, in uniform neighbourhoods, so they are easier than the average cell and these values should be read as an upper bound on how well the map matches its own label source.

![**Figure 3.** Overall agreement with the map-agreement labels under random folds and 5, 10 and 20 km spatial blocks. Keeping the plantation class makes agreement fall once whole areas are held out; in the final scheme it barely changes. This is agreement with labels from existing maps, not map accuracy.](paper/figures/fig_cv.jpg)

## 3.3 The 2025 map

![**Figure 4.** Land cover in 2025 from the random forest. Forest includes tree plantations and may include some tea.](paper/figures/fig2_landcover.jpg)

Table 2. Mapped area by class in 2025 (cell counts × 0.0004 km²). These are mapped areas, not estimates of the true area; the reference sample is needed for those.

{{area_table}}

The mean top-class probability of the random forest is {{prob_forest}} for forest and {{prob_built}} for built-up cells. Built-up land is the least certain class (Figure 4) and the one where the maps differ most.

## 3.4 Comparison with WorldCover, Esri and Hansen

![**Figure 5.** Agreement of the 2025 map with WorldCover 2021 and Esri 2023 after translation into the same classes.](paper/figures/fig3_agreement.jpg)

WorldCover and the Esri map disagree on {{dis_km2}} km² of the box. On these cells, which the training sample left out, the 2025 map sides with the Esri map in {{side_es_pct}}% of cells, with WorldCover in {{side_wc_pct}}% and with neither in {{side_neither_pct}}%. Over the whole box, the 2025 map agrees with WorldCover on {{agree_wc_pct}}%, with the Esri map on {{agree_es_pct}}%, with both on {{agree_both_pct}}% and with neither on {{agree_neither_pct}}%; the two existing maps agree with each other on {{wc_es_pct}}%. Because the training labels are the cells where these two maps agree, this agreement is partly built in and is not evidence of accuracy. The largest single disagreement is {{top1_km2}} km² that WorldCover calls tree cover and both the Esri map and this map call cropland and grassland; next is {{top2_km2}} km² that this map and WorldCover call forest and the Esri map calls crops or rangeland. In Figure 5 the disagreements lie mainly in the farmed areas around the forest blocks, where small fields, hedgerows, woodlots and tea mix within a few cells. Forest area is {{wc_forest_km2}} km² in WorldCover, {{forest_km2}} km² here and {{es_forest_km2}} km² in the Esri map; built-up land is {{wc_built_km2}}, {{built_km2}} and {{es_built_km2}} km² (Figure 6).

![**Figure 6.** Mapped area by class in ESA WorldCover 2021, this map and Esri 2023, from pixel counts after translation into the same classes. These are mapped areas, not estimates with confidence intervals.](paper/figures/fig_areas.jpg)

![**Figure 7.** Tree cover loss by year inside the study box in Hansen Global Forest Change v1.12. Loss includes plantation harvest and is not deforestation alone.](paper/figures/fig4_hansen.jpg)

Hansen records {{loss_total_km2}} km² of tree cover loss inside the box from 2001 to 2024 (Figure 7); for scale, {{tc2000_km2}} km² of the box had at least 30% tree cover in 2000. The largest yearly totals are {{loss_max_km2}} km² in {{loss_max_year}} and {{loss_second_km2}} km² in {{loss_second_year}}; the smallest is {{loss_min_km2}} km² in {{loss_min_year}}. Of the cells with loss at any time in 2001–2024, {{loss_in_forest_pct}}% are forest in the 2025 map, which could reflect regrowth, replanting or map error, and this study cannot tell which; the rest are mapped as cropland and grassland or other classes. Of the cells mapped as forest in 2025, {{forest_tc2000_pct}}% had at least 30% tree cover in 2000.

## 3.5 Reference sites from 2015

At {{lesiv_match}} of the {{lesiv_sites}} Geo-Wiki sites from 2015 inside the box, the most common class in the reference square is also the most common class of the 2025 map in the same 100 m square. The sites were not drawn from this map, are mostly cropland, and are ten years older, so this is not an accuracy estimate.

## 3.6 Accuracy and area estimates

{{accuracy_section}}

# 4. Discussion

With labels from two existing maps, the dry-season composite separates forest, open farmland and grassland, built-up land and water well enough that blocked and random cross-validation give similar results. The plantation class is the exception: its samples come from a few clusters, and when whole blocks are held out the classifier calls most of them forest. That is the result the blocked test was meant to expose. Random folds suggested a plantation recall of {{c_plant_rec_random}}; blocked folds give {{plant_rec}}. Merging the classes keeps the final map honest about what it can show, at the cost of not separating natural forest from plantations, which is often the distinction that matters in the Mau.

The small gap between random and blocked agreement in the final scheme should not be read as proof that spatial autocorrelation is harmless here. The training cells are uniform patches on which two maps agree, the classes left are spectrally distinct in the dry season, and the blocks have no buffer. The test that would show how the map performs on the hard cells, the mixed farm and woodland mosaics where the maps disagree, is the independent reference sample, which is still to be labelled.

The comparison maps disagree most in the farm mosaic around the forest. WorldCover counts more of it as tree cover and the Esri map counts more as crops and built area. This map, trained on cells where they agree, falls between them: {{forest_wc_pct}}% of its forest cells are tree cover in WorldCover against {{forest_es_pct}}% in the Esri map, and {{crop_es_pct}}% of its cropland and grassland cells are crops or rangeland in the Esri map against {{crop_wc_pct}}% in WorldCover. This agreement is partly built in, as noted above. Which map is closer to the ground cannot be settled by comparing maps.

# 5. Limitations

There are no field data. Training labels come from WorldCover 2021, Esri 2023 and a 2015 forest management map, so their errors, definitions and dates carry into the map, and the uniform cells used for training under-represent the mixed cells where errors are most likely. Cross-validated agreement is agreement with those labels, not map accuracy. The training sample is balanced by class (forest {{train_forest_pct}}% of training cells, built-up {{train_built_pct}}%), unlike the landscape (built-up {{built_pct}}% of the mapped box), which can push the classifier to over-map rare classes: built-up land is {{built_km2}} km² here against {{wc_built_km2}} km² in WorldCover. The composite keeps cells that the scene classification calls unclassified, and no buffer was put around clouds or shadows, so some cloud edges may remain in the median. The spatial blocks have no buffer between training and test cells, so some leakage across block edges remains. The composite covers one dry season, January to March 2025, so seasonal crops, fallow and grass look alike, and the class "cropland and grassland" joins them. With the available plantation labels, plantations could not be separated from natural forest; the forest class includes them, and some tea may be mapped as forest. No bare class was trained, because the two maps agree on almost no bare cells in the box. Built-up land is the least certain class. Hansen loss is a separate product with a different tree definition, and its yearly values are affected by changes in processing over the record. The 21 reference sites from 2015 are only a descriptive check. The study area is a rectangle, not the legal boundary of the forest reserves, so the areas here are not areas of the gazetted forest. The reference sample, once labelled, will rest on one interpreter reading high-resolution imagery of uncertain date, not on field visits. We did not test block sizes other than 5, 10 and 20 km or other classifiers, and did not tune either model.

# 6. Conclusion

A random forest trained on map-agreement labels maps forest, cropland and grassland, built-up land and water around the Mau Forest Complex in 2025 with cross-validated agreement of {{f_rf_block10}} against those labels under 10 km spatial blocks. With the available plantation labels ({{plant_share_pct}}% of the box in the 2015 100 m map), plantations could not be separated from natural forest under spatial blocking, so they are mapped together. The map shows {{forest_km2}} km² of forest in the {{box_km2}} km² box, between the WorldCover and Esri figures. {{conclusion_accuracy}}

# Data and code availability

Code (MIT licence) and derived data (CC BY 4.0) are available from <https://github.com/oloyededavid/Mau-Forest-Land-Cover>. Every result in this paper is read from the files in `outputs/` that the scripts write. Source data keep their own terms: Copernicus Sentinel-2 and DEM, ESA WorldCover (CC BY 4.0), Esri / Impact Observatory land cover (CC BY 4.0), Hansen Global Forest Change (CC BY 4.0), the forest management map and the Geo-Wiki reference data (CC BY 4.0).

# References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cochran, W. G. (1977). *Sampling techniques* (3rd ed.). Wiley.

Drusch, M., Del Bello, U., Carlier, S., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment*, 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026

ESA. (2022). *Copernicus DEM: Copernicus Digital Elevation Model*. European Space Agency. https://doi.org/10.5270/ESA-c5d3d65

Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Fritz, S., See, L., Perger, C., et al. (2016). *A global dataset of crowdsourced land cover and land use reference data (2011–2012)*. PANGAEA. https://doi.org/10.1594/PANGAEA.869682

Fritz, S., See, L., Perger, C., et al. (2017). A global dataset of crowdsourced land cover and land use reference data. *Scientific Data*, 4, 170075. https://doi.org/10.1038/sdata.2017.75

Hansen, M. C., Potapov, P. V., Moore, R., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853. https://doi.org/10.1126/science.1244693

Karra, K., Kontgis, C., Statman-Weil, Z., Mazzariello, J. C., Mathis, M., & Brumby, S. P. (2021). Global land use / land cover with Sentinel 2 and deep learning. *2021 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 4704–4707. https://doi.org/10.1109/IGARSS47720.2021.9553499

Kweyu, R. M., Thenya, T., Kiemo, K., & Emborg, J. (2020). The nexus between land cover changes, politics and conflict in Eastern Mau forest complex, Kenya. *Applied Geography*, 114, 102115. https://doi.org/10.1016/j.apgeog.2019.102115

Lesiv, M., Schepaschenko, D., Buchhorn, M., et al. (2022). Global forest management data for 2015 at a 100 m resolution. *Scientific Data*, 9, 199. https://doi.org/10.1038/s41597-022-01332-3

Lesiv, M., Fritz, S., Dürauer, M., et al. (2025a). A global reference data set for land cover mapping at 10 m resolution. *Earth System Science Data*, 17(11), 6149–6155. https://doi.org/10.5194/essd-17-6149-2025

Lesiv, M. (2025b). *Global land cover data set at 10m for 2015 (Geo-Wiki)*. Zenodo. https://doi.org/10.5281/zenodo.14871660

Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42–57. https://doi.org/10.1016/j.rse.2014.02.015

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Ploton, P., Mortier, F., Réjou-Méchain, M., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540. https://doi.org/10.1038/s41467-020-18321-y

Radoux, J., Lamarche, C., Van Bogaert, E., Bontemps, S., Brockmann, C., & Defourny, P. (2014). Automated training sample extraction for global land cover mapping. *Remote Sensing*, 6(5), 3965–3987. https://doi.org/10.3390/rs6053965

Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

Stehman, S. V., & Foody, G. M. (2019). Key issues in rigorous accuracy assessment of land cover products. *Remote Sensing of Environment*, 231, 111199. https://doi.org/10.1016/j.rse.2019.05.018

Zanaga, D., Van De Kerchove, R., Daems, D., et al. (2022). *ESA WorldCover 10 m 2021 v200*. Zenodo. https://doi.org/10.5281/zenodo.7254221

Zhang, X., Liu, L., Chen, X., Gao, Y., Xie, S., & Mi, J. (2021). GLC_FCS30: global land-cover product with fine classification system at 30 m using time-series Landsat imagery. *Earth System Science Data*, 13(6), 2753–2776. https://doi.org/10.5194/essd-13-2753-2021
