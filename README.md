# ADTGAN MODEL

## Abstract: 

In this study, we present  the development and application of a novel Generative Adversarial Network (GAN) model in the oceanography. ADTGAN(Absolute Dynamic Topography Generated by Generative Adversarial Network(GAN)), was designed to generate absolute dynamic topography (SSH) data and reconstruct geostrophic currents in the Mediterranean Sea. Using a comprehensive data set from the Copernicus Marine Environmental Monitoring Service, which spanned 7 May 2001 to 5 July 2021, we focused on the components of absolute dynamic topography (SSH) and time information to train our model.


The ADTGAN model was evaluated under two configurations: a simple structure without ResNet and Attention layers and a complex structure that incorporates these layers. In addition, we examine the impact of including temporal information in model inputs. Our results indicate that the inclusion of temporal data significantly improves model accuracy and generalization. The simple model structure achieved a mean square error (MSE) of $0.00214$ m, while the complex structure provided $0.008$ m. Incorporating temporal information further lowered the MSE to $0.0014$ m for the simple model and $0.0016$ m for the complex model respectively.

Validation involved comparing the synthetic absolute dynamic topography(SSH) data and the derived geostrophic currents with in situ observations. The complex model with temporal information demonstrated the most accurate and realistic geostrophic current patterns, closely matching observed dynamics in the Mediterranean Sea, including key features such as Tyrrhenian Sea gyres, Alboran Sea gyres, southern Adriatic anticyclonic circulation, northern current, levantine cyclonic circulation, and Algerian current.

Our findings highlight the importance of model complexity and temporal information in accurately capturing the dynamic nature of absolute dynamic topography(SSH) and geostrophic currents. The ADTGAN model outperforms traditional methods, demonstrating significant potential for enhancing SSH data and geostrophic current reconstruction. Future research should focus on refining GAN architectures, validating synthetic SSH data against in-situ measurements, and exploring the broader applicability of ADTGAN to other oceanographic regions.
