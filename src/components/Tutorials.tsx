import React from "react";
import { Link } from "react-router-dom";
import Typography from "@material-ui/core/Typography";
import Grid from "@material-ui/core/Grid";

import ng_contrast from "./ng_contrast.png";
import ng_resolution from "./ng_resolution.png";

export default function Tutorials() {
  return (
    <div style={{ maxWidth: "45em", marginLeft: "auto", marginRight: "auto" }}>
      <Typography variant="h3" gutterBottom>
        OpenOrganelle Tutorials
      </Typography>
      <Typography variant="h4" gutterBottom>
        Website
      </Typography>
      <Typography variant="h5">Navigating OpenOrganelle</Typography>
      <Typography>
        Our data portal is organized on five main pages. Datasets, Software,
        Tutorials, Publications, and Organelles. In these pages you will find
        tools and resources for browsing and consuming the data. For more
        information about the mission of the data portal and the teams involved
        please visit our About page.
      </Typography>
      <Typography variant="h5">Datasets</Typography>
      <Typography>
        On the <Link to="/">Datasets</Link> page you will find a list of all the
        available FIB-SEM datasets and accompanying segmentations. Clicking on a
        dataset will direct you to that individual dataset’s page where you can
        see more detailed information about image acquisition, data location,
        links for viewing in FIJI. On the individual dataset page you can also
        select layers to view in Neuroglancer to browse the data online. More
        about the layer selection can be found in <a href="">Visualization</a>.
      </Typography>
      <Typography variant="h5">Software</Typography>
      <Typography>
        On the <Link to="/software">Software</Link> page is a comprehensive list
        of all the software that was used to generate these datasets. This
        includes commercially available and homewritten software used for
        manual, machine learning libraries, analysis, and much more. Here you
        will also find useful links to our GitHub repositories.
      </Typography>
      <Typography variant="h5">Tutorials</Typography>
      <Typography>
        On the <Link to="/tutorials">Tutorials</Link> page, this page, is a
        compilation of instructions for navigating OpenOrganelle.
      </Typography>
      <Typography variant="h5">Publications</Typography>
      <Typography>
        On the <Link to="/publications">Publications</Link> page you’ll find
        publications associated with these datasets as well as a list of “views”
        that are referenced in these publications. More information about
        pre-made data views can be found in
        <a href="">Visualization</a>.
      </Typography>
      <Typography variant="h5">Organelles</Typography>
      <Typography>
        On the <Link to="/organelles">Organelles</Link> page is a catalog of all
        of the organelles that have been segmented. Included here is a
        description of the organelles as well as Neurglancer links to examples
        of these organelles in four different datasets.
      </Typography>
      <Typography variant="h4" gutterBottom>
        Visualization
      </Typography>
      <Typography>
        On each dataset page are customizable options to browse the data in a
        web based viewer, Neuroglancer. Customizations include:
      </Typography>
      <ul>
        <li>
          Pre-made “views”: These views are pre-determined regions of interest.
          Some are referenced in related publications, others are just
          interesting and of note! For a catalog of views that are directly
          related to published material please visit the “Publications” page.
        </li>
        <li>
          “Layers”: the FIB-SEM volumes, available segmentations, and light
          microscopy volumes are each a layer that can be added to the
          Neuroglancer instance.
        </li>
      </ul>
      <Typography variant="h5">How to setup a Neuroglancer instance</Typography>
      <Typography>On a dataset page- </Typography>
      <ol>
        <li>
          First, select a “view”. Either a pre-made view or the default location
          provided.
        </li>
        <li>
          Next, select the layers you would like to have visible in the
          Neuroglancer instance. Available options may include FIB-SEM,
          organelle predictions, refined organelle segmentations, analysis of
          organelle segmentations (i.e. contact sites, skeletons, and
          curvature), and correlative light microscopy.
        </li>
        <li>
          Once the view and layers are selected, you can initiate a Neuroglancer
          instance in a new tab by clicking the “VIEW” button or the dataset
          thumbnail.
        </li>
      </ol>
      <Typography variant="h5">How to use Neuroglancer</Typography>
      <Typography variant="h6">Keyboard and mouse bindings</Typography>
      <Typography>
        Below is a non-exhaustive list of useful keyboard strokes and mouse
        clicks for browsing data within Neuroglancer. For a complete set of
        bindings, within Neuroglancer, press h or click on the button labeled ?
        in the upper right corner.
      </Typography>
      <ul>
        <li> Left Click on a layer name to toggle its visibility.</li>
        <li>
          Right Click on a layer to select it and modify its settings in the
          side panel.
        </li>
        <li>Space to toggle the view layout.</li>
        <li>Left Click and drag the data to pan.</li>
        <li>Right Click to recenter at that location.</li>
        <li>Mouse Wheel to scroll through 2D slices.</li>
        <li>Shft + Wheel to scroll at 10x speed.</li>
        <li>Ctrl + Wheel to zoom.</li>
        <li>Shift + Left Click to rotate in plane.</li>
        <li>z to return to an orthogonal plane.</li>
        <li>Double Left Click a segmentation to turn on/off a 3D rendering.</li>
        <li>a to toggle axis lines</li>
        <li>b to toggle scale bar</li>
      </ul>
      <Typography variant="h6">Other useful tools In the side panel</Typography>
      <Typography>
        (Ctrl + Left Click to show if closed) are customizable settings to
        optimize the viewing experience of the data.
      </Typography>
      <Typography variant="h6">Side Panel - Source</Typography>
      <Typography>On this tab you can find:</Typography>
      <ul>
        <li>The data location on S3.</li>
        <li>The data type.</li>
        <li>The voxel size.</li>
      </ul>
      <Typography>
        To manually add a new layer, click the + button, select the data type
        (ex. N5), and enter the data location (ex. s3://…).
      </Typography>
      <Typography variant="h6">Side Panel - Rendering</Typography>
      <ul>
        <li>
          <img
            src={ng_resolution}
            alt="Resoultion controls from neuroglancer"
            style={{ float: "right", margin: "0.5em" }}
          />
          Resolution (slice): The data quality vs load speed can be customized.
          For instance, to optimize for quality (slow loading times) click on
          the left side of the histogram. To reset the settings, simply double
          click the histogram.
        </li>
        <li>
          Blending: Select the preferred blending of the selected layer, i.e.
          default or additive.
        </li>
        <li>Opacity: Select the preferred opacity of the selected layer.</li>
        <li>
          <img
            src={ng_contrast}
            alt="Contrast controls from neuroglancer"
            style={{ float: "right", margin: "0.5em" }}
          />
          Contrast: Adjust the contrast of the selected layer. To invert the
          lookup table click the arrow. To adjust the contrast click and drag
          the lower and upper bound on the graph.
        </li>
        <li>
          Color: For segmentation layers, a preferred color can be selected by
          clicking the color box.
        </li>
      </ul>
      <Typography variant="h6">Side Panel - Seg.</Typography>
      <Typography>
        If 3D renderings are available for a segmentation they will be visible
        on the segmentation panel. As you double click segmentations to turn on
        3D renderings, their ID will become visible here. Selecting/deselecting
        renderings can also be performed by clicking on the IDs in the panel.
      </Typography>
      <Typography variant="h6">Top Toolbar - Coordinates</Typography>
      <Typography>
        In the top left hand corner you will find the XYZ coordinates (in nm).
        To copy a location you can click the copy icon. To navigate to a
        specific location, you can manually enter in the XYZ coordinates.
      </Typography>
      <Typography variant="h4" gutterBottom>
        Data handling tutorials
      </Typography>
      <Typography variant="h5">Data organization</Typography>
      <Typography>
        General data and metadata structure? - See our{" "}
        <a href="https://github.com/janelia-cosem/schemas/blob/master/README.md">
          schema
        </a>{" "}
        documentation.
      </Typography>
      <ul>
        <li>Raw EM data</li>
        <li>LM</li>
        <li>Predictions</li>
        <li>Segmentations</li>
        <li>
          Analysis (curvature, skeletons, contact sites, classified
          segmentations)
        </li>
      </ul>
      <Typography>
        You can find an extensive list of available predictions, segmentations,
        and analysis on the <Link to="/organelles">Organelles</Link> page.
      </Typography>
      <Typography variant="h5">How to open data in Fiji</Typography>
      <Typography>
        There are a couple options to view and/or open our N5 data within Fiji.
        Within each individual dataset page you will find a Fiji icon . Clicking
        the icon will copy the dataset location to use in the Fiji app.
      </Typography>
      <Typography variant="h6">Viewing data</Typography>
      <Typography>
        To simply view and browse the data, and take advantage of its
        multi-scale properties, use the BigDataViewer Plugin, Plugins &rarr;
        BigDataViewer &rarr; N5 Viewer. For instructions refer to N5 Viewer. See
        the BigDataviewer page for details for navigation / interaction
        documentation.
      </Typography>
      <Typography variant="h6">Opening data</Typography>
      <Typography>
        To open the datasets in Fiji and make use of all of the Fiji tools, File
        &rarr; Import &rarr; N5. For instructions on opening our datasets in
        Fiji please refer to n5-ij. Note, this option currently does not support
        multi-scale functionality.
      </Typography>
      <Typography variant="h5">Downloading data</Typography>
      <Typography>
        Software page? point people to python / command line tools for
        downloading stuff from s3 AWS command line tools for downloading
      </Typography>
      <Typography variant="h5">How to access analysis database(s)</Typography>
      <Typography>
        The code used to process predictions and perform analysis can be found
        here: https://github.com/janelia-cosem/hot-knife/tree/cosem-analysis.
        Evaluations and metrics
      </Typography>

      <Typography variant="h4" gutterBottom>
        Sharing
      </Typography>
      <Typography>
        We invite you to share and use this data broadly! The data is licensed
        under <a href="https://creativecommons.org/licenses/by/4.0/legalcode">CC BY 4.0</a>. You are free to share and adapt this data. We ask that
        you please be sure to cite the data DOIs and the related publication(s).
        All of this information can be found listed on the individual data page.
        If you are redistributing the data, please link back to OpenOrganelle.
      </Typography>

      <Typography>
        There are multiple avenues to share the datasets. We recommend sharing
        individual dataset pages (i.e.
        https://openorganelle.janelia.org/datasets/jrc_hela-2) so others can
        easily customize their Neuroglancer environments, find the data location
        on S3, and find related material available (i.e. metadata, analysis,
        segmentations, light microscopy). For consistency, we also recommend
        referring to the data using their Data IDs (i.e. jrc_hela-2).
      </Typography>

      <Typography>
        We welcome you to visit our GitHub repository to access all of our code
        and software: https://github.com/janelia-cosem. More information about
        the software used and written for this project can be found on the
        “Software” page.
      </Typography>

      <Typography>
        For inquiries about contributing to this platform please contact:{" "}
        <a href="mailto:cosemdata@janelia.hhmi.org">
          cosemdata@janelia.hhmi.org
        </a>
      </Typography>
    </div>
  );
}
