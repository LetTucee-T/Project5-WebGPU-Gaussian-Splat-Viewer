export function decodeHeader(plyArrayBuffer: ArrayBuffer): [number, Record<string, string>, DataView] {
    /* decodes the .ply file header and returns a tuple of:
        * - vertexCount: number of vertices in the point cloud
        * - propertyTypes: a map from property names to their types
        * - vertexData: a DataView of the vertex data
    */

    const decoder = new TextDecoder();
    const bytes = new Uint8Array(plyArrayBuffer);
    const headerLines: string[] = [];
    let lineStart = 0;
    let vertexByteOffset = -1;
    // Byte offsets keep CRLF and UTF-8 comments from shifting the binary body.
    for (let i = 0; i < Math.min(bytes.length, 1024 * 1024); ++i) {
        if (bytes[i] !== 10) continue;
        const line = decoder.decode(bytes.subarray(lineStart, i)).replace(/\r$/, '');
        headerLines.push(line);
        lineStart = i + 1;
        if (line === 'end_header') { vertexByteOffset = i + 1; break; }
    }
    if (headerLines[0] !== 'ply' || vertexByteOffset < 0) throw new Error('Invalid or incomplete PLY header.');
    if (!headerLines.includes('format binary_little_endian 1.0')) {
        throw new Error('This viewer expects a binary_little_endian 1.0 Gaussian PLY file.');
    }

    let vertexCount = 0;
    let propertyTypes: Record<string, string> = Object.create(null);
    let inVertices = false;
    let sawVertices = false;
    const scalarBytes: Record<string, number> = { float: 4, float32: 4, uchar: 1, uint8: 1 };

    for (let i = 0; i < headerLines.length; i++) {
        const line = headerLines[i].trim();
        const fields = line.split(/\s+/);
        if (fields[0] === 'element') {
            inVertices = fields[1] === 'vertex';
            if (inVertices) {
                if (sawVertices) throw new Error('Duplicate PLY vertex element.');
                sawVertices = true;
                vertexCount = Number(fields[2]);
            } else if (!sawVertices && Number(fields[2]) !== 0) {
                throw new Error('The vertex element must be the first nonempty PLY element.');
            }
        } else if (inVertices && fields[0] === 'property') {
            const [, type, name] = fields;
            if (!Object.prototype.hasOwnProperty.call(scalarBytes, type) || fields.length !== 3) throw new Error(`Unsupported PLY property: ${line}`);
            if (Object.prototype.hasOwnProperty.call(propertyTypes, name)) throw new Error(`Duplicate PLY property: ${name}`);
            propertyTypes[name] = type;
        } else if (line === 'end_header') {
            break;
        }
    }

    if (!Number.isSafeInteger(vertexCount) || vertexCount < 1) throw new Error('PLY must contain at least one vertex.');
    const stride = Object.values(propertyTypes).reduce((sum, type) => sum + scalarBytes[type], 0);
    if (!stride || vertexCount > Math.floor((plyArrayBuffer.byteLength - vertexByteOffset) / stride)) throw new Error('PLY vertex data is truncated.');
    const vertexData = new DataView(plyArrayBuffer, vertexByteOffset, vertexCount * stride);

    return [
        vertexCount,
        propertyTypes,
        vertexData,
    ];
}

export function readRawVertex(offset: number, vertexData: DataView, propertyTypes: Record<string, string>): [number, Record<string, number>] {
    /* reads a single vertex from the vertexData DataView and returns a tuple of:
        * - offset: the offset of the next vertex in the vertexData DataView
        * - rawVertex: a map from property names to their values
    */
    let rawVertex: Record<string, number> = Object.create(null);

    for (const property in propertyTypes) {
        const propertyType = propertyTypes[property];
        if (propertyType === 'float' || propertyType === 'float32') {
            rawVertex[property] = vertexData.getFloat32(offset, true);
            offset += Float32Array.BYTES_PER_ELEMENT;
        } else if (propertyType === 'uchar' || propertyType === 'uint8') {
            rawVertex[property] = vertexData.getUint8(offset) / 255.0;
            offset += Uint8Array.BYTES_PER_ELEMENT;
        }
    }

    return [offset, rawVertex];
}

export function nShCoeffs(sphericalHarmonicsDegree: number): number {
    /* returns the expected number of spherical harmonics coefficients */
    if (sphericalHarmonicsDegree === 0) {
        return 1;
    } else if (sphericalHarmonicsDegree === 1) {
        return 4;
    } else if (sphericalHarmonicsDegree === 2) {
        return 9;
    } else if (sphericalHarmonicsDegree === 3) {
        return 16;
    } else {
        throw new Error(`Unsupported SH degree: ${sphericalHarmonicsDegree}`);
    }
}
