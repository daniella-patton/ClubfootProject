# ClubfootProject<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Clubfoot Project</h3>
<p align="center">
  <a href="https://github.research.chop.edu/pattondm/Clubfoot-Project">
    <img src="/Clubfoot_im.png" alt="Logo" width="500" height="400">
  </a>
   <p align="center">
    Deep learning project for automating angle measures of patients suspected of of clubfoot.
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#Prerequisites">Prerequisites</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

**Key Objective:** Develop a deep learning algorithm that can calculate 5 angles and one displacement measurement used to diagnose club foot severities and types match the measurements calculated by a radiologist and orthopedist expert.


**Secondary Objectives:**  Decrease the overall time required to make key measurements to assess clubfoot severity compared to that of an experienced radiologist. Provided a more objective, consistent, measures to assess clubfoot severity by evaluating the inter-observer agreement between three experienced radiologist. 
Proposed workflow: 
* Identify the key bones necessary to make calculations using a deep learning algorithm. This will require identification of 4 bones in the AP direction and four in the lateral. 
* Use basic computations to automated the angle measures (partial PCA, geometrical calculation, etc.). 

<!-- PREREQUISITES -->
## Prerequisites
* Pandas 1.1.3
* Matplotlib 3.3.2
* Torch 1.7.1
* Numpy 1.19.2
* SimpleITK 2.0.2
* PIL 8.0.1
* Torchvision 0.8.2
* Scipy 1.5.2
* Cv2 4.0.1

Additional Packages:
os, time, copy, gc, random, glob



<!-- ROADMAP -->
## Roadmap

SECTION IN PROGRESS

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. 
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. 



<!-- CONTACT -->
## Contact

Daniella Patton - pattondm@chop.edu

Project Link: [https://github.research.chop.edu/pattondm/Clubfoot-Project](https://github.research.chop.edu/pattondm/Clubfoot-Project)





<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
Tehcnical expertise was provided by Dr. Michael Francavilla, Dr. Susan Sotardi, Dr. Adarsh Ghosh, Dr. Minhhui, and Dr. Hao Huang
Clinical epertise was provided by Dr. Raymond Sze and Dr. Jie Nguyan


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: LICENSE.txt
