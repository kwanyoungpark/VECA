import struct,json

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
    status_code = decode(recvall(conn, 1), 'uint8'); print("Status", status_code)
    if use_metadata: metadata_length = decode(recvall(conn, 4), 'int'); print("MDL:", metadata_length)
    length = decode(recvall(conn, 4), 'int'); print("Len:", length)
    if use_metadata: payload_metadata = json.loads(decode(recvall(conn, metadata_length), 'str')) ; print("MD", payload_metadata)
    payload_json = decode(recvall(conn, length), 'str') ; print("JSON", payload_json)
    payload = json.loads(payload_json)
    if use_metadata: return status_code, metadata_length, length, payload_metadata, payload
    else: return status_code, length, payload

# STATUS CODE
class STATUS:
    STEP = 100
    REST = 101
    INIT = 102
    RECO = 200
    CLOS = 201
    

# terminator
terminator = b'\xFF\xFE\xFE\xFF',
