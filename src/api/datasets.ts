import {
  urlSafeStringify,
  encodeFragment,
  CoordinateSpaceTransform,
  CoordinateSpace,
  ViewerState,
  ImageLayer,
  LayerDataSource,
  SegmentationLayer,
  ManagedLayer,
} from "@janelia-cosem/neuroglancer-url-tools";
import { s3ls, getObjectFromJSON, bucketNameToURL } from "./datasources";
import * as Path from "path";
import {DatasetDescription} from "./dataset_description"

const meshContainerProtocol = 'precomputed'
const IMAGE_DTYPES = ['int8', 'uint8', 'uint16'];
const SEGMENTATION_DTYPES = ['uint64'];
type LayerTypes = 'image' | 'segmentation' | 'annotation' | 'mesh';

type VolumeStores = "n5" | "precomputed" | "zarr";
export type ContentType = "em" | "segmentation";

interface ContrastLimits {
  min: number
  max: number
}

interface DisplaySettings {
  contrastLimits: ContrastLimits;
  gamma: number;
  invertColormap: boolean;
  color: string;
  defaultLayerType: LayerTypes
}

export class DatasetView {
    constructor(
    public name: string, 
    public description: string, 
    public volumeKeys: string[],
    public position?: number,          
    public scale?: number){
        
        this.name = name;
        this.description = description;

        if (position === null) {this.position = undefined}
        else (this.position = position)

        if (scale === null) {this.scale = undefined}
        else (this.scale = scale)

        this.volumeKeys = volumeKeys;
    }
  }

  export interface MeshSource {
    path: string
    name: string
    datasetName: string
    format: string
  }

interface DatasetIndex {
  name: string;
  volumes: VolumeSource[];
  meshes: MeshSource[]
  views: DatasetView[]
}

interface SpatialTransform {
  axes: string[];
  units: string[];
  translate: number[];
  scale: number[];
}

interface N5PixelResolution {
  dimensions: number[];
  unit: string;
}

interface N5ArrayAttrs {
    name: string
    dimensions: number[]
    dataType: string
    pixelResolution: N5PixelResolution
    offset?: number[]
}

interface NeuroglancerPrecomputedScaleAttrs {
  chunk_sizes: number[];
  encoding: string;
  key: string;
  resolution: number[];
  size: number[];
  voxel_offset: number[];
  jpeg_quality?: number;
}

interface NeuroglancerPrecomputedAttrs {
    at_type: string
    data_type: string
    type: string
    scales: NeuroglancerPrecomputedScaleAttrs[]
    num_channels: number
}

const displayDefaults: DisplaySettings = {
  contrastLimits: {min: 0, max:1},
  invertColormap: false,
  color: "white",
};

// one nanometer, expressed as a scaled meter
const nm: [number, string] = [1e-9, "m"];

// this specifies the basis vectors of the coordinate space neuroglancer will use for displaying all the data
const outputDimensions: CoordinateSpace = { x: nm, y: nm, z: nm };

const defaultView = new DatasetView('Default view', '', [], undefined, undefined);

function makeShader(shaderArgs: DisplaySettings, contentType: ContentType, dataType: string): string | undefined{
  let lower = 0;
  let upper = 0;
  let cmin = 0;
  let cmax = 0;  
    switch (contentType) {
    case 'em':{
      if (dataType === 'uint8') {lower = 0; upper = 255}
      else if (dataType === 'uint16') {lower = 0; upper = 65535}
      let cmin = shaderArgs.contrastLimits.min * (upper - lower);
      let cmax = shaderArgs.contrastLimits.max * (upper - lower);
        return `#uicontrol invlerp normalized(range=[${cmin}, ${cmax}], window=[${Math.max(0, cmin - 2 * (cmax-cmin))}, ${Math.min(upper, cmax + 2 * (cmax - cmin))}])
        #uicontrol int invertColormap slider(min=0, max=1, step=1, default=${shaderArgs.invertColormap? 1: 0})
        #uicontrol vec3 color color(default="${shaderArgs.color}") 
        float inverter(float val, int invert) {return 0.5 + ((2.0 * (-float(invert) + 0.5)) * (val - 0.5));}
          void main() {
          emitRGB(color * inverter(normalized(), invertColormap));
        }`
      };
    case "segmentation":
      return `#uicontrol invlerp normalized(range=[${cmin}, ${cmax}], window=[${cmin}, ${cmax}])\n#uicontrol vec3 color color(default="${shaderArgs.color}")\nvoid main() {emitRGB(color * normalized());}`;
      default:
        return undefined;
              }
}

// A single n-dimensional array
// this constructor syntax can be shortened but I forget how
export class VolumeSource {
    constructor(
        public path: string,
        public name: string,
        public datasetName: string,
        public dataType: string,
        public dimensions: number[],
        public transform: SpatialTransform,
        public contentType: ContentType,
        public containerType: VolumeStores,
        public displaySettings: DisplaySettings,
        public description: string,
        public version: string,
        public tags: string[],
    ) {
        this.path = path;
        this.name = name;
        this.datasetName = datasetName;
        this.dataType = dataType;
        this.dimensions = dimensions;
        this.transform = transform;
        this.contentType = contentType;
        this.containerType = containerType;
        this.displaySettings = displaySettings;
        this.description = description;
        this.version = version;
        this.tags = tags;
     }

    // todo: remove handling of spatial metadata, or at least don't pass it on to the neuroglancer
    // viewer state construction

    toLayer(layerType: LayerTypes, mesh: Optional[MeshSource]): ImageLayer | SegmentationLayer {
        const srcURL = `${this.containerType}://${this.path}`;
        let meshURL = mesh ? `${meshContainerProtocol}://${mesh.path}` : undefined;

        const inputDimensions: CoordinateSpace = {
            x: [1e-9 * this.transform.scale[this.transform.axes.indexOf('x')], "m"],
            y: [1e-9 * this.transform.scale[this.transform.axes.indexOf('y')], "m"],
            z: [1e-9 * this.transform.scale[this.transform.axes.indexOf('z')], "m"]
        };
        const layerTransform: CoordinateSpaceTransform = {matrix:
            [
                [1, 0, 0, this.transform.translate[this.transform.axes.indexOf('x')]],
                [0, 1, 0, this.transform.translate[this.transform.axes.indexOf('y')]],
                [0, 0, 1, this.transform.translate[this.transform.axes.indexOf('z')]]
            ],
            outputDimensions: outputDimensions,
            inputDimensions: inputDimensions}
        // need to update the layerdatasource object to have a transform property
        
        const source = meshURL? [{url: srcURL, transform: layerTransform}, meshURL] : [{url: srcURL, transform: layerTransform}];
        
        let layer = undefined;

        if (layerType === 'image'){
          let shader: string | undefined = '';
          if (SEGMENTATION_DTYPES.includes(this.dataType)){
              shader = makeShader(this.displaySettings, 'segmentation', this.datasetName);
          }
          else if (IMAGE_DTYPES.includes(this.dataType)) {
              shader = makeShader(this.displaySettings, 'em', this.dataType);
          }
          else {shader = undefined}
          layer = new ImageLayer('rendering',
                                undefined,
                                undefined,
                                source,
                                0.75,
                                'additive',
                                shader,
                                undefined,
                                undefined);
        }
        else if (layerType === 'segmentation') {
            layer = new SegmentationLayer(source, 'source', true);
        }
        
        layer.name = this.name
        return layer
    }
}

// a collection of volumes, i.e. a collection of ndimensional arrays
export class Dataset {
    public key: string;
    public space: CoordinateSpace;
    public volumes: Map<string, VolumeSource>;
    public meshes: Map<string, MeshSource>
    public description: DatasetDescription | undefined
    public thumbnailPath: string
    public views: DatasetView[]
    constructor(key: string, 
               space: CoordinateSpace, 
               volumes: Map<string, VolumeSource>,
               meshes: Map<string, MeshSource>, 
               description: DatasetDescription,
               thumbnailPath: string, 
               views: DatasetView[]) {
        this.key = key;
        this.space = space;
        this.volumes = volumes;
        this.meshes = meshes;
        this.description = description;
        this.thumbnailPath = thumbnailPath;
        this.views = views;
    }

    makeNeuroglancerViewerState(view: DatasetView): string {        
        const layers = [...this.volumes.keys()].filter(a => view.volumeKeys.includes(a)).map(a => 
          {let vol = this.volumes.get(a);
            let mesh = this.meshes.get(a);
            return vol.toLayer(vol.displaySettings.defaultLayerType, mesh)});       
        // hack to post-hoc adjust alpha if there is only 1 layer selected
        if (layers.length  === 1) {layers[0].opacity = 1.0}
        const viewerPosition = view.position;
        let crossSectionScale = view.scale ? view.scale : 50.0;
        const projectionOrientation = undefined;
        const crossSectionOrientation = undefined;
        const projectionScale = 65536;
         // the first layer is the selected layer; consider making this a kwarg
        const selectedLayer = {'layer': layers[0].name, 'visible': true};

        const vState = new ViewerState(
            outputDimensions,
            viewerPosition,
            layers,
            '4panel',
            undefined,
            crossSectionScale,
            undefined,
            crossSectionOrientation,
            projectionScale,
            undefined,
            projectionOrientation,
            true,
            true,
            true,
            undefined,
            undefined,
            undefined,
            undefined,
            'black',
            'black',
            selectedLayer,
            undefined
        );
        return encodeFragment(urlSafeStringify(vState));
    }    
  }


async function getDatasetKeys(bucket: string): Promise<string[]> {
    // get all the folders in the bucket
    let datasetKeys = (await s3ls(bucket, '', '/', '', false)).folders;
    //remove trailing "/" character
    datasetKeys = datasetKeys.map((k) => k.replace(/\/$/, ""));
    return datasetKeys
}

async function getDatasetIndex(
  bucket: string,
  datasetKey: string
): Promise<DatasetIndex> {
  const bucketURL = bucketNameToURL(bucket);
  const indexFile = `${bucketURL}/${datasetKey}/index.test.json`;
  return getObjectFromJSON(indexFile);
}

async function getDescription(
  bucket: string,
  key: string
): Promise<DatasetDescription> {
  const bucketURL = bucketNameToURL(bucket);
  const descriptionURL = `${bucketURL}/${key}/README.json`;
  return getObjectFromJSON(descriptionURL);
}

function reifyPath(outerPath: string, innerPath: string): string {
  const absPath = new URL(outerPath);
  absPath.pathname = Path.resolve(absPath.pathname, innerPath);
  return absPath.toString();
}

function makeMeshSource(outerPath: string, meshMeta: MeshSource): MeshSource {
  meshMeta.path = reifyPath(outerPath, meshMeta.path);
  return meshMeta
}

function makeVolumeSource(outerPath: string, volumeMeta: VolumeSource): VolumeSource {
  volumeMeta.path = reifyPath(outerPath, volumeMeta.path);
  // this is a shim until we add a defaultLayerType field to the volume metadata
  let ds = volumeMeta.displaySettings;
  if (volumeMeta.dataType === 'uint64' && (volumeMeta.name.indexOf('_seg') === -1)) {ds.defaultLayerType = 'segmentation'}
  else (ds.defaultLayerType = 'image')
  volumeMeta.displaySettings = ds;

  // console.log([volumeMeta.name, volumeMeta.displaySettings.defaultLayerType])
  // this looks so stupid! there must be a better way to do this that doesn't enrage the 
  // linter
  return new VolumeSource(volumeMeta.path, 
                    volumeMeta.name, 
                    volumeMeta.datasetName, 
                    volumeMeta.dataType,
                    volumeMeta.dimensions,
                    volumeMeta.transform, 
                    volumeMeta.contentType,
                    volumeMeta.containerType,
                    volumeMeta.displaySettings,
                    volumeMeta.description,
                    volumeMeta.version,
                    volumeMeta.tags)
}

export async function makeDatasets(bucket: string): Promise<Map<string, Dataset>> {
    // get the keys to the datasets
    const datasetKeys: string[] = await getDatasetKeys(bucket);
    // Get a list of volume metadata specifications, represented instances of Map<string, VolumeMeta>
    const datasets: Map<string, Dataset> = new Map();
    for (const key of datasetKeys) {
        const outerPath: string = `${bucketNameToURL(bucket)}/${key}`;
        const description = await getDescription(bucket, key);
        const thumbnailPath: string =  `${outerPath}/thumbnail.jpg`
        const index = await getDatasetIndex(bucket, key);
        if (index !== undefined){
            try {
            const views: DatasetView[] = [];
            // make sure that the default view is at the beginning of the list
            for (let v of index.views) {
              let vObj = new DatasetView(v.name, v.description, v.volumeKeys, v.position, v.scale);
              if (vObj.name === 'Default View'){views.unshift(vObj)}
              else views.push(vObj)
            }
            if (views.length === 0){views.push(defaultView)}
            const volumes: Map<string, VolumeSource> = new Map();
            const meshes: Map<string, MeshSource> = new Map()
            index.volumes.forEach(v => volumes.set(v.name, makeVolumeSource(outerPath, v)));
            index.meshes.forEach(v => meshes.set(v.name, makeMeshSource(outerPath, v)));
            console.log(meshes);
            datasets.set(key, new Dataset(key, outputDimensions, volumes, meshes, description, thumbnailPath, views));
        }
        catch (error) {
            console.log(error)
        }
        }
        else {console.log(`Could not load index.json from ${outerPath}`)}
    }
    return datasets
}
