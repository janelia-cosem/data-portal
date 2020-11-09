import React from "react";
import Typography from "@material-ui/core/Typography";

export default function About() {
  return (
    <div className="content">
      <Typography
        variant="h3"
        gutterBottom
        style={{ marginTop: "2em", textAlign: "center" }}
      >
        Welcome to HHMI Janelia’s OpenOrganelle.
      </Typography>
      <div
        style={{ maxWidth: "45em", marginLeft: "auto", marginRight: "auto" }}
      >
        <p style={{ textIndent: "3ch" }}>
          On this data portal we present large volume, high resolution
          3D-Electron Microscopy (EM) datasets, acquired with the enhanced
          focused ion beam scanning electron microscopy (FIB-SEM) technology
          developed at Janelia (
          <a href="https://elifesciences.org/articles/25916">Xu et al. 2017</a>,{" "}
          <a href="https://patents.google.com/patent/US10600615B2/en">
            Xu et al. 2020a
          </a>
          ,{" "}
          <a href="https://link.springer.com/protocol/10.1007/978-1-0716-0691-9_12">
            Xu et al. 2020b
          </a>
          ). Accompanying these EM volumes are automated segmentations and
          analysis of intracellular sub-structures made possible by the COSEM
          Project Team (Heinrich et al., 2020).
        </p>

        <p style={{ textIndent: "3ch" }}>
          Within Janelia are some of the world’s leading experts in Machine
          Learning (
          <a href="https://www.janelia.org/lab/saalfeld-lab">Saalfeld lab</a>{" "}
          and <a href="https://www.janelia.org/lab/funke-lab">Funke lab</a>),
          Cell Biology (
          <a href="https://www.janelia.org/lab/lippincott-schwartz-lab">
            Lippincott-Schwartz lab
          </a>
          ), and large-volume, high-resolution EM data acquisition (
          <a href="https://www.janelia.org/lab/hess-lab">Hess lab</a> and{" "}
          <a href="https://www.janelia.org/lab/fib-sem-technology">
            FIB-SEM Technology
          </a>
          ). This provides a unique opportunity for COSEM to expand research
          that lies at the intersection of those fields and drive forward our
          tools to study and knowledge about subcellular structures.{" "}
        </p>

        <p style={{ textIndent: "3ch" }}>
          The unprecedented volume and resolution of EM datasets generated by
          the enhanced FIB-SEM platform (FIB-SEM Technology and Hess Lab) both
          demands and enables the development of universal machine learning
          classifiers for the automatic detection of sub-cellular structures in
          these data. The COSEM team, in collaboration with the Saalfeld and
          Funke labs with biological guidance from the Lippincott-Schwartz lab,
          has begun tackling the segmentation problem as well as the subsequent
          large-scale quantitative analysis of these data to answer biological
          questions.{" "}
        </p>

        <p style={{ textIndent: "3ch" }}>
          On this site you will find all datasets, training data and
          segmentations available for online viewing and download. Be sure to
          also check out our tutorials to learn how to work with the data
          yourself and our publications list to learn more!
        </p>

        <Typography variant="h5">Acknowledgements:</Typography>

        <p style={{ textIndent: "3ch" }}>
          This work is part of the COSEM Project Team at Janelia Research
          Campus, Howard Hughes Medical Institute, Ashburn, VA. During this
          effort, the COSEM Project Team consisted of: Riasat Ali, Rebecca
          Arruda, Rohit Bahtra, Davis Bennett, Destiny Nguyen, Woohyun Park, and
          Alyson Petruncio, led by Aubrey Weigel, with Steering Committee of Jan
          Funke, Harald Hess, Wyatt Korff, Jenniffer Lippincott-Schwartz, and
          Stephan Saalfeld. We thank Rohit Bahtra, a Janelia-LCR Summer
          Internship Program student, for his work generating masks of datasets
          and providing manual annotations. We thank Arslan Aziz, a Janelia
          Undergraduate Scholars Program student, for his work correcting
          mitochondria over-merging. We thank Gudrun Ihrke and Project Technical
          Resources for management and coordination and staff support. We thank
          Janelia Scientific Computing Shared Resource, especially Tom Dolafi
          and Stuart Berg. We thank Victoria Custard for administrative support.
          We thank Song Pang and Gleb Shtengel for their work collecting and
          organizing the FIB-SEM data to seed OpenOrganelle. We further thank
          Gleb Shtengel for his early work manually segmenting organelles,
          motivating the need for more automated approaches. We thank K.
          Hayworth and W. Qiu at Howard Hughes Medical Institute (HHMI) Janelia
          Research Campus (JRC) for invaluable discussions and data collection
          support. We gratefully acknowledge P. Rivlin, S. Plaza, and I.
          Meinertzhagen for JRC EM shared resource and FlyEM project team
          support on staining protocols development. We thank the electron
          microscopy facility of MPI-CBG and of the CMCB Technology Platform at
          TU Dresden for their services. We also thank Y. Wu from Pietro De
          Camilli’s laboratory at Yale for advice.
        </p>

        <p>
          Many of the datasets used in this research were derived from a HeLa
          cell line. Henrietta Lacks, and the HeLa cell line that was
          established from her tumor cells without her knowledge or consent in
          1951, have made significant contributions to scientific progress and
          advances in human health. We are grateful to Henrietta Lacks, now
          deceased, and to her surviving family members for their contributions
          to biomedical research.
        </p>

        <p>
          {" "}
          We especially thank Amazon Web Services for free hosting of our data
          through their open data program.
        </p>

        <p>
          Funding provided by{" "}
          <a href="https://hhmi.org">Howard Hughes Medical Institute</a>
        </p>
      </div>
    </div>
  );
}
