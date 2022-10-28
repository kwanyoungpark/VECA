import struct,json
from typing import Tuple
import struct, base64
import numpy as np


def decode(byteArray, numtype):
    if numtype == 'uint8':
        size = len(byteArray)
        return struct.unpack('B' * size, byteArray)[0]
    elif numtype == 'int16':
        size = len(byteArray) // 2
        return struct.unpack('h' * size, byteArray)[0]
    elif numtype == 'int':
        size = len(byteArray) // 4
        return struct.unpack('i' * size, byteArray)[0]
    elif numtype == 'float':
        size = len(byteArray) // 4
        return struct.unpack('f' * size, byteArray)[0]
    elif numtype == 'uint8[]':
        size = len(byteArray)
        return struct.unpack('B' * size, byteArray)
    elif numtype == 'int16[]':
        size = len(byteArray) // 2
        return struct.unpack('h' * size, byteArray)
    elif numtype == 'int[]':
        size = len(byteArray) // 4
        return struct.unpack('i' * size, byteArray)
    elif numtype == 'float[]':
        size = len(byteArray) // 4
        return struct.unpack('f' * size, byteArray)
    elif numtype == 'str':
        return byteArray.decode('ascii')
    elif numtype == 'char[]':
        return byteArray.decode('utf-16')
    else:
        raise NotImplementedError()

def recvall(socket, length):
    remainder = length
    buffer = b''
    while remainder > 0:
        message = socket.recv(min(4096, remainder))
        buffer += message
        remainder -= len(message)
    return buffer

types = ['char', 'int', 'float', 'uint8', 'int16']
typesz = [2, 4, 4, 1, 2]

def build_packet(status, outputs):
    # input : int(status), list of bytearrays
    # output : bytearray
    packet_l = packet_format(status,outputs)
 
    # list of bytes to a single bytearray
    from functools import reduce
    bytelen = reduce(lambda acc, x: acc + len(x), packet_l, 0)
    packet = bytearray(bytelen)
    count = 0
    for elem in packet_l :
        packet[count:count+len(elem)] = elem
        count += len(elem)
    #print("Len packet : ",len(packet))
    #print("len box coord : ", len(packet[1:]))
    return packet

def packet_format(status, outputs):
    assert all(isinstance(element,bytes) for element in outputs)
    outputs.insert(0,status.to_bytes(1,'little'))
    return outputs

def build_json_packet(status, payload, metadata = None, use_metadata = False):
    assert isinstance(payload,dict)
    assert not (use_metadata and (metadata is None))
    payload_json = json.dumps(payload).encode('utf-8')
    if use_metadata:
        metadata_json = json.dumps(metadata).encode('utf-8')
        return build_packet(status, [len(metadata_json).to_bytes(4, 'little'), len(payload_json).to_bytes(4, 'little'),metadata_json, payload_json])
    else: return build_packet(status, [len(payload_json).to_bytes(4, 'little'),payload_json])

def recv_json_packet(conn, use_metadata = False):
    status_code = decode(recvall(conn, 1), 'uint8')
    if use_metadata: metadata_length = decode(recvall(conn, 4), 'int')
    length = decode(recvall(conn, 4), 'int')
    if use_metadata: payload_metadata = json.loads(decode(recvall(conn, metadata_length), 'str'))
    payload_json = decode(recvall(conn, length), 'str')
    payload = json.loads(payload_json)
    if use_metadata: return status_code, metadata_length, length, payload_metadata, payload
    else: return status_code, length, payload

def _protocol_encode(data:dict):
    output = {}
    metadata = {}
    def _encode(x) -> Tuple[str,str]:
        info = str(type(x))
        base64_ascii = lambda t:  base64.b64encode(t).decode("ascii")
        if isinstance(x,np.ndarray):
            return "/".join([str(type(x)), str(x.dtype), str(x.shape)]), base64_ascii(x.tobytes())
        elif isinstance(x, list):
            y = np.array(x)
            return "/".join([str(type(y)), str(y.dtype), str(y.shape)]), base64_ascii(y.tobytes())
        elif isinstance(x, str):
            return info, x
        elif isinstance(x, int):
            return info, base64_ascii(np.array([x]))
        elif isinstance(x, float):
            return info, base64_ascii(np.array([x]))
        elif isinstance(x, bytes):
            return info, base64_ascii(x)
        else:
            print(type(x))
            print(x.dtype)
            raise NotImplementedError()
    for key, value in data.items():
        type_enc, value_enc = _encode(value)
        output[key] = value_enc
        metadata[key] = type_enc

    return metadata, output
def _protocol_decode(metadata:dict, data:dict):
    def _decode(x, info) : 
        info = info.split("/")
        typeinfo = info[0]
        print("Info:", info)
        if "str" in typeinfo:
            return x
        elif "int" in typeinfo:
            return x[0]
        elif any([t in typeinfo for t in [ "float", "bytes"]]):
            return base64.b64decode(x[0].encode('ascii'))
        elif typeinfo in ["System.Int32", "System.Int64","System.Float32", "System.Float64"] :
            return x
        shape = tuple(int(x) for x in info[-1].replace("(","").replace(",)","").replace(")","").split(","))
        if typeinfo == "System.Byte[]":
            return np.frombuffer(base64.b64decode(x[0].encode('ascii')), np.uint8).reshape(shape)
        elif typeinfo == "System.Int16[]":
            return np.array(x).reshape(shape)
        elif typeinfo == "System.Single[]":
            return np.array(x).reshape(shape)
        elif "numpy.ndarray" in typeinfo:
            if isinstance(x, list):
                return np.array(x)
            else:
                bytedata = base64.b64decode(x.encode('ascii'))
                if "uint8" in info[1]:
                    nptype = np.uint8
                elif "float32" in info[1]:
                    nptype = np.float32 
                elif "float64" in info[1]:
                    nptype = np.float64
                elif "int32" in info[1]:
                    nptype = np.int32
                elif "int64" in info[1]:
                    nptype = np.int64
                else:
                    print("Type in:", info[1])
                    raise NotImplementedError()
                return np.frombuffer(bytedata, nptype).reshape(shape)
        else:
            print(typeinfo)
            raise NotImplementedError()
    return {key: _decode(value,metadata[key]) for key,value in data.items()}

def request(conn, status, data:dict):
    metadata, data = _protocol_encode(data)
    packet = build_json_packet(status, data, metadata, use_metadata= True)
    conn.sendall(packet)

def response(conn):
    status, _, _, metadata, data = recv_json_packet(conn, use_metadata = True)
    return status, metadata, _protocol_decode(metadata,data)

# STATUS CODE
class STATUS:
    STEP = 100
    REST = 101
    INIT = 102
    RECO = 200
    CLOS = 201
    

# terminator
terminator = b'\xFF\xFE\xFE\xFF',
