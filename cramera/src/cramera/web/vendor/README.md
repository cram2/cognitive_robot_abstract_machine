# Browser dependencies

These browser distributions are shipped locally. The viewer makes no CDN requests.
`checksums.json` records the SHA-256 of every distributed JavaScript file.

| Files | Release and source | License |
| --- | --- | --- |
| `three.min.js`, `RoomEnvironment.js`, `ColladaLoader.js`, `GLTFLoader.js`, `MTLLoader.js`, `OBJLoader.js`, `OrbitControls.js`, `STLLoader.js`, `CopyShader.js`, `SSAOShader.js`, `SimplexNoise.js`, `EffectComposer.js`, `RenderPass.js`, `ShaderPass.js`, `SSAOPass.js` | [Three.js 0.128.0](https://github.com/mrdoob/three.js/tree/r128), npm `three@0.128.0` | [MIT](licenses/three-MIT.txt) |
| `URDFLoader.js` | [urdf-loader 0.12.1](https://github.com/gkjohnson/urdf-loaders/tree/8f83b645499d486ca2bbc41b81b8de9d0a3b3f4d), npm `urdf-loader@0.12.1`, `umd/URDFLoader.js` | [Apache-2.0](licenses/urdf-loader-Apache-2.0.txt) |
| `vis-network.min.js` | [vis-network 9.1.9](https://github.com/visjs/vis-network/tree/v9.1.9), npm `vis-network@9.1.9`, `standalone/umd/vis-network.min.js` | [MIT](licenses/vis-network-MIT.txt), chosen from its dual license |

Every listed JavaScript file was compared byte for byte with the corresponding npm
release archive. Three.js example files come from `examples/js/`; its main bundle
comes from `build/three.min.js`. Their existing copyright headers are preserved.

URDF loader copyright: Copyright © 2020 California Institute of Technology.
ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
Neither the name of Caltech nor its operating division, the Jet Propulsion
Laboratory, nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission.

The URDF loader license was obtained from the exact upstream release commit,
because that npm archive omits the license file. Three.js and vis-network license
texts come directly from their release archives. All license texts above are
included in the installed wheel.
