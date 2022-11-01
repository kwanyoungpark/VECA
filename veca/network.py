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


str_npType = {
    "byte":np.int8, "char":np.uint16, "short":np.int16, "int":np.int32, "long":np.int64, "float":np.float32, "double":np.float64
}
npType_str = {np.dtype(k):v for (k, v) in {
    np.int8: "byte", np.uint8: "byte",
    np.int16: "short", np.uint16: "char", 
    np.int32: "int", 
    np.int64: "long", 
    np.float32: "float", 
    np.float64: "double"
}.items()}
Primitive_str = {
    int: "int", float: "float"
}
str_Primitive = {
    "int": int, "byte": lambda x: np.array([x],np.uint8), "float": float
}

base64_ascii_encode = lambda t:  base64.b64encode(t).decode("ascii")
base64_ascii_decode = lambda t:  base64.b64decode(t.encode('ascii'))

def _encode(x) -> Tuple[str,str]:
    
    if isinstance(x,str):
        return "string", x
    elif isinstance(x, list):
        if all([isinstance(e, str) for e in x]):
            return "string[]", "#".join(x)
    elif isinstance(x,np.ndarray):
        return "/".join(["array", npType_str[x.dtype], str(x.shape)]),  base64_ascii_encode(x.tobytes())
    elif any([isinstance(x, t) for t in [int, float]]):
        return "primitive/" + Primitive_str[type(x)] , base64_ascii_encode(np.array([x]).tobytes())
    else:
        raise NotImplementedError("Not Implemented for type" + str(type(x)))
    '''
    elif ctype == "string[]":
        return x.split('#')
    elif ctype == "array":
        dtype = str_npType[api[1]]
        shape_str = api[2].replace("(","").replace(",)","").replace(")","").split(",")
        shape = tuple(int(x) for x in shape_str)
        return np.frombuffer(x.encode('utf8'), dtype).reshape(shape)
    elif ctype == "primitive":
        dtype = str_npType[api[1]]
        return np.frombuffer(x.encode('utf8'), dtype)[0]
    else:
        raise NotImplementedError("Not implemented for type:" + ctype)
    info = str(type(x))
    
    if isinstance(x,np.ndarray):
        return "/".join([str(type(x)), str(x.dtype), str(x.shape)]), x.tobytes().decode('utf-8')
    elif isinstance(x, list):
        if isinstance(x[0], str):
            return "strings", x
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
        #print(x.dtype)
        raise NotImplementedError()
    '''

def _protocol_encode(data:dict):
    output = {}
    metadata = {}
    
    for key, value in data.items():
        type_enc, value_enc = _encode(value)
        output[key] = value_enc
        metadata[key] = type_enc

    return metadata, output

def _decode(x, api) : 
    api = api.split("/")
    ctype = api[0]
    if ctype == "string":
        return x
    elif ctype == "string[]":
        return x.split('#')
    elif ctype == "array":
        dtype = str_npType[api[1]]
        shape_str = api[2].replace("(","").replace(",)","").replace(")","").split(",")
        shape = tuple(int(x) for x in shape_str)
        return np.frombuffer(base64_ascii_decode(x), dtype).reshape(shape)
    elif ctype == "primitive":
        dtype = str_npType[api[1]]
        return str_Primitive[api[1]](np.frombuffer(base64_ascii_decode(x), dtype)[0])
    else:
        raise NotImplementedError("Not implemented for type:" + ctype)

def _protocol_decode(metadata:dict, data:dict):
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
    HELP = 202

class HelpException(Exception):
    pass

# terminator
terminator = b'\xFF\xFE\xFE\xFF',
