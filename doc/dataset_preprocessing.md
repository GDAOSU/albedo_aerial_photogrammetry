#### Data Collection

We collected datasets using a DJI Phantom 4 Pro drone. The drone was flown at an altitude of 50 meters above the ground level. The camera was set to capture images at a resolution of 5472 x 3648 pixels. The images were captured in RAW+JPEG format.

#### Data Preparation

The classic photogrammetry processing is done by Bentley ContextCapture, including aerial triangulation, dense matching, and 3D reconstruction. 

For image poses, we export `Blocks Exchange` format.

``` xml
<?xml version="1.0" encoding="utf-8"?>
<BlocksExchange version="3.2">
  <SpatialReferenceSystems>
    <SRS>
      <Id>0</Id>
      <Name>WGS 84 - World Geodetic System 1984 (EPSG:4326) + EGM96 geoid height (EPSG:5773)</Name>
      <Definition>EPSG:4326+5773</Definition>
    </SRS>
    <SRS>
      <Id>2</Id>
      <Name>Local East-North-Up (ENU); origin: 39.975230N 83.006780W</Name>
      <Definition>ENU:39.97523,-83.00678</Definition>
    </SRS>
  </SpatialReferenceSystems>
  <Block>
    <Name>Block_1 - AT</Name>
    <Description>Result of aerotriangulation of Block_1 (2022-Mar-25 17:20:06)</Description>
    <SRSId>2</SRSId>
    <Photogroups>
      <Photogroup>
        <Name>Photogroup 1</Name>
        <ImageDimensions>
          <Width>4856</Width>
          <Height>3640</Height>
        </ImageDimensions>
        <CameraModelType>Perspective</CameraModelType>
        <CameraModelBand>Visible</CameraModelBand>
        <FocalLength>9.9001356603859</FocalLength>
        <SensorSize>13.2</SensorSize>
        <CameraOrientation>XRightYDown</CameraOrientation>
        <PrincipalPoint>
          <x>2406.3622927091</x>
          <y>1856.42166463931</y>
        </PrincipalPoint>
        <AspectRatio>1</AspectRatio>
        <Skew>0</Skew>
        <Photo>
          <Id>0</Id>
          <ImagePath>D:/CC_Workspace/GoodalePark_UAV/Rectify/Block_1 - AT - Rect - ENU undistorted photos/DJI_0001.jpg</ImagePath>
          <Component>1</Component>
          <Pose>
            <Rotation>
              <M_00>-0.234602658410435</M_00>
              <M_01>0.971058900239015</M_01>
              <M_02>0.0447906790900752</M_02>
              <M_10>0.437332200289725</M_10>
              <M_11>0.146583148425574</M_11>
              <M_12>-0.887273310309397</M_12>
              <M_20>-0.868160203681617</M_20>
              <M_21>-0.188568271096278</M_21>
              <M_22>-0.45906412175126</M_22>
              <Accurate>true</Accurate>
            </Rotation>
            <Center>
              <x>29.9350296852194</x>
              <y>4.36444716951105</y>
              <z>127.390985712561</z>
              <Accurate>true</Accurate>
            </Center>
            <Metadata>
              <SRSId>0</SRSId>
              <Rotation>
                <M_00>-0.21473532751176</M_00>
                <M_01>0.976672278258382</M_01>
                <M_02>9.0396468088727e-11</M_02>
                <M_10>0.503023409651369</M_10>
                <M_11>0.110596869696836</M_11>
                <M_12>-0.857167300913872</M_12>
                <M_20>-0.837171540642136</M_20>
                <M_21>-0.18406410104864</M_21>
                <M_22>-0.515038074557627</M_22>
                <Accurate>false</Accurate>
              </Rotation>
              <Center>
                <x>-83.0064163888889</x>
                <y>39.9752701666667</y>
                <z>157.918</z>
                <Accurate>true</Accurate>
              </Center>
            </Metadata>
          </Pose>
          <NearDepth>10.1860120454541</NearDepth>
          <MedianDepth>23.8322188168539</MedianDepth>
          <FarDepth>78.7552925659252</FarDepth>
          <ExifData>
            <ImageDimensions>
              <Width>4856</Width>
              <Height>3640</Height>
            </ImageDimensions>
            <GPS>
              <Latitude>39.9752701666667</Latitude>
              <Longitude>-83.0064163888889</Longitude>
              <Altitude>157.918</Altitude>
            </GPS>
            <FocalLength>8.8</FocalLength>
            <FocalLength35mmEq>24</FocalLength35mmEq>
            <Make>DJI</Make>
            <Model>FC6310S</Model>
            <DateTimeOriginal>2022-01-04T18:25:49</DateTimeOriginal>
          </ExifData>
          <ColorModel>
            <Blockwise>0,0.23967544889414466,0.00019429617983353166,0.85775561050324201,0.62532760410262289,1,0,0.24337784895431602,0.027307623579387194,0.7753952288620195,0.65091527682096395,1,0,0.2149425395667495,0.092670788546924093,0.76198020174675241,0.58908407186934986,1</Blockwise>
          </ColorModel>
        </Photo>
      </Photogroup>
    </Photogroups>
    <ControlPoints/>
    <PositioningConstraints/>
    <ColorModelReference>
      <Blockwise>0,0.14409215268276937,0.47626492564507222,0.26432939677260764,0.83287480052322393,1,0,0.16977756858444112,0.42498275813457415,0.3268194382303008,0.82048013806079667,1,0,0.16254884374184725,0.43507946518307711,0.33765551815322614,0.82416033682835821,1</Blockwise>
    </ColorModelReference>
  </Block>
</BlocksExchange>
```

For 3D model production, we use local ENU coordinate system for model reconstruction. `Tiling system` was setted to regular tiling. The output model is in OBJ format. Make sure each tile has at most 1 texture image. To be safe, you can export obj with no texture.

``` bash
metadata.xml
Data
├── Tile_+000_+000
  ├── Tile_+000_+000.obj
  ├── Tile_+000_+000.mtl
  ├── Tile_+000_+000.jpg
  ├── Tile_+000_+000.xml
├── Tile_+000_+001
  .
  .
  .
```

`metadata.xml` file contains the following information:

``` xml
<?xml version="1.0" encoding="utf-8"?>
<ModelMetadata version="1">
	<!--Spatial Reference System-->
	<SRS>ENU:39.97523,-83.00678</SRS>
	<!--Origin in Spatial Reference System-->
	<SRSOrigin>0,0,0</SRSOrigin>
	<Texture>
		<ColorSource>Visible</ColorSource>
	</Texture>
</ModelMetadata>
```