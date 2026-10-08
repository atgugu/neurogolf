#!/usr/bin/env python3
"""task133: proven geometry front + byte/QLinearMatMul render relower.

Rule: recover the complete 3x3 creature from the unique unmagnified sprite,
then complete each partial magnified sprite with its signature/body colors.
"""
from pathlib import Path
import hashlib
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'refs'/'kaggle_proven_best.onnx'; OUT=ROOT/'task133.onnx'
SHA='87d9324f157ed7e9b16295f937e16a3555eea41a53eb0eb74a28c69d75cc3ac6'
N=helper.make_node

def add_init(m,name,a):
    m.graph.initializer.append(numpy_helper.from_array(np.asarray(a),name))

def qmat_relower(m):
    """Replace the unsupported integer Einsum idea with supported QLinearMatMul."""
    add_init(m,'qscale',np.array(1,np.float32)); add_init(m,'qzero',np.array(0,np.uint8))
    add_init(m,'sh309',np.array([30,9],np.int64)); add_init(m,'sh930',np.array([9,30],np.int64))
    add_init(m,'yvR',np.arange(30,dtype=np.float16).reshape(30,1,1))
    add_init(m,'xvC',np.arange(30,dtype=np.float16).reshape(1,1,30))
    nn=[]
    for n in m.graph.node:
        o=n.output[0] if n.output else ''
        if o=='pat33f': continue
        if o=='hiR':
            nn += [n,N('Transpose',['loR'],['loR_ysr'],perm=[1,0,2]),N('Transpose',['hiR'],['hiR_ysr'],perm=[1,0,2]),N('GreaterOrEqual',['yvR','loR_ysr'],['geR']),N('Less',['yvR','hiR_ysr'],['ltR']),N('And',['geR','ltR'],['Rb']),N('Cast',['Rb'],['R'],to=TensorProto.UINT8)]
        elif o in {'geR','ltR','Rb','R'}: continue
        elif o=='hiC':
            nn += [n,N('Transpose',['loC'],['loC_scx'],perm=[0,2,1]),N('Transpose',['hiC'],['hiC_scx'],perm=[0,2,1]),N('GreaterOrEqual',['xvC','loC_scx'],['geC']),N('Less',['xvC','hiC_scx'],['ltC']),N('And',['geC','ltC'],['Cb']),N('Cast',['Cb'],['CmT'],to=TensorProto.UINT8)]
        elif o in {'geC','ltC','Cb','Cm'}: continue
        elif o=='Vcell':
            nn += [N('Cast',['Sc'],['Sc_u8'],to=TensorProto.UINT8),N('Mul',['pat33','Sc_u8'],['Vcell'])]
        elif o=='bodyf':
            nn += [N('QLinearMatMul',['Vcell','qscale','qzero','CmT','qscale','qzero','qscale','qzero'],['Bq']),N('Reshape',['R','sh309'],['Rflat']),N('Reshape',['Bq','sh930'],['Bflat']),N('QLinearMatMul',['Rflat','qscale','qzero','Bflat','qscale','qzero','qscale','qzero'],['body_u8'])]
        elif o=='body_u8': continue
        else: nn.append(n)
    del m.graph.node[:];m.graph.node.extend(nn)
    return m

def optimize(m):
    add_init(m,'sh19',np.array([1,9],np.int64));add_init(m,'yvR2',np.arange(30,dtype=np.int8).reshape(30,1))
    add_init(m,'one_u8',np.array(1,np.uint8));add_init(m,'two_i8',np.array(2,np.int8))
    add_init(m,'syl_i8',np.array([[[[-1,1,0,0]]]],np.int8));add_init(m,'sxl_i8',np.array([[[[0,0,-1,1]]]],np.int8))
    add_init(m,'z_i8',np.array(0,np.int8));add_init(m,'n29_i8',np.array(29,np.int8))
    add_init(m,'i_row_i8',np.array([2,1,0,-1],np.int8));add_init(m,'i_col_i8',np.array([3,2,1,0,-1,-2],np.int8))
    add_init(m,'syl_col_i8',np.array([-1,1,0,0],np.int8).reshape(4,1));add_init(m,'sxl_col_i8',np.array([0,0,-1,1],np.int8).reshape(4,1))
    for i in m.graph.initializer:
        if i.name=='cds': i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.uint8),'cds'))
        if i.name=='xvC': i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.int8),'xvC'))
    nn=[]
    dead={'cnt_eq1','cnt_eq4','cnt_eq9','cnt_eq16','cnt_sq14','cnt_sq916','cnt_square','cnt0f','nz','mf','Mcode','issqf','Pcode'}
    for n in m.graph.node:
        o=n.output[0] if n.output else ''
        if o in dead: continue
        if o=='H_f32': nn += [n,N('Floor',['H_f32'],['H_floor']),N('Equal',['H_f32','H_floor'],['cnt_square'])]
        elif o=='loR_ysr': nn.append(N('Reshape',['loR','sh19'],['loR_ysr']))
        elif o=='hiR_ysr': nn.append(N('Reshape',['hiR','sh19'],['hiR_ysr']))
        elif o in {'geR','ltR'}: nn.append(N(n.op_type,['yvR2',n.input[1]],list(n.output)))
        elif o=='Rflat': continue
        elif o=='body_u8': n.input[0]='R';nn.append(n)
        elif o=='mcds': nn += [N('Cast',['mb'],['mb_u8'],to=TensorProto.UINT8),N('ArgMax',['mb_u8'],['Midx'],axis=1,keepdims=1),N('Cast',['Midx'],['Midx_u8'],to=TensorProto.UINT8),N('Add',['Midx_u8','one_u8'],['Mcode_u8'])]
        elif o=='Mcode_u8': continue
        elif o=='H': nn += [n,N('Cast',['H'],['H_i8'],to=TensorProto.INT8)]
        elif o=='topf': nn += [n,N('Cast',['topf'],['top_i8'],to=TensorProto.INT8)]
        elif o=='leftf': nn += [n,N('Cast',['leftf'],['left_i8'],to=TensorProto.INT8)]
        elif o=='Hh': nn.append(N('Div',['H_i8','two_i8'],['h2']))
        elif o=='h2': continue
        elif o=='sylH': nn.append(N('Mul',['H_i8','syl_i8'],['sylH']))
        elif o=='sxlH': nn.append(N('Mul',['H_i8','sxl_i8'],['sxlH']))
        elif o=='r4a': nn.append(N('Add',['top_i8','sylH'],['r4a']))
        elif o=='c4a': nn.append(N('Add',['left_i8','sxlH'],['c4a']))
        elif o in {'r4b','c4b'}: nn.append(N('Add',[n.input[0],'h2'],list(n.output)))
        elif o=='r4': nn += [N('Clip',['r4b','z_i8','n29_i8'],['r4']),N('Cast',['r4'],['r4_f16'],to=TensorProto.FLOAT16)]
        elif o=='c4': nn += [N('Clip',['c4b','z_i8','n29_i8'],['c4']),N('Cast',['c4'],['c4_f16'],to=TensorProto.FLOAT16)]
        elif o=='r30': nn.append(N('Mul',['r4_f16','c30'],['r30']))
        elif o=='flatf': nn.append(N('Add',['r30','c4_f16'],['flatf']))
        elif o=='hitf': nn.append(N('Cast',['hit'],['hit_u8'],to=TensorProto.UINT8))
        elif o=='anyhit': nn += [N('ReduceMax',['hit_u8','ax3'],['anyhit_u8'],keepdims=1),N('Cast',['anyhit_u8'],['anyhit_b'],to=TensorProto.BOOL)]
        elif o=='hsyl': nn += [N('MatMulInteger',['hit_u8','syl_col_i8'],['sy_i32']),N('Cast',['sy_i32'],['sy'],to=TensorProto.INT8)]
        elif o=='hsxl': nn += [N('MatMulInteger',['hit_u8','sxl_col_i8'],['sx_i32']),N('Cast',['sx_i32'],['sx'],to=TensorProto.INT8)]
        elif o in {'sy','sx'}: continue
        elif o=='shp0_pre': nn.append(N('And',['solid_b','anyhit_b'],['shp0_pre']))
        elif o=='onemmf': nn.append(N('Not',['mb'],['onemmf']))
        elif o=='shp0': nn.append(N('And',['shp0_pre','onemmf'],['shp0']))
        elif o=='scode': nn += [N('Where',['shp0','cds','qzero'],['scode_u8']),N('Cast',['scode_u8'],['scode'],to=TensorProto.FLOAT16)]
        elif o=='oneshp': nn.append(N('Not',['shp0'],['oneshp']))
        elif o=='pb0': nn.append(N('And',['present_b','oneshp'],['pb0']))
        elif o=='pbm': nn.append(N('And',['pb0','onemmf'],['pbm']))
        elif o=='pcds': nn += [N('Cast',['pbm'],['pbm_u8'],to=TensorProto.UINT8),N('ArgMax',['pbm_u8'],['Pidx'],axis=1,keepdims=1),N('Cast',['Pidx'],['Pidx_u8'],to=TensorProto.UINT8),N('Add',['Pidx_u8','one_u8'],['Pcode_u8'])]
        elif o=='Pcode_u8': continue
        elif o=='ptop': nn += [N('Cast',['pbm'],['pbm_f16'],to=TensorProto.FLOAT16),N('Mul',['pbm_f16','topf'],['ptop'])]
        elif o=='pleft': nn.append(N('Mul',['pbm_f16','leftf'],['pleft']))
        elif o=='syH': nn.append(N('Mul',['sy','H_i8'],['syH']))
        elif o=='sxH': nn.append(N('Mul',['sx','H_i8'],['sxH']))
        elif o=='mr0': nn.append(N('Add',['top_i8','syH'],['mr0']))
        elif o=='mc0': nn.append(N('Add',['left_i8','sxH'],['mc0']))
        elif o=='mr': nn.append(N('Clip',['mr0','z_i8','n29_i8'],['mr']))
        elif o=='mc': nn.append(N('Clip',['mc0','z_i8','n29_i8'],['mc']))
        elif o=='mrf': nn.append(N('Reshape',['mr','sh10'],['mrf']))
        elif o=='mcf': nn.append(N('Reshape',['mc','sh10'],['mcf']))
        elif o=='Hf': nn.append(N('Reshape',['H_i8','sh10'],['Hf']))
        elif o=='roff': nn.append(N('Gather',['i_row_i8','ridx'],['roff'],axis=0))
        elif o=='coff': nn.append(N('Gather',['i_col_i8','cidx'],['coff'],axis=0))
        else: nn.append(n)
    del m.graph.node[:];m.graph.node.extend(nn);del m.graph.value_info[:]
    used={x for n in m.graph.node for x in n.input if x}; keep=[x for x in m.graph.initializer if x.name in used]
    del m.graph.initializer[:];m.graph.initializer.extend(keep)
    return m

def main():
    if hashlib.sha256(SRC.read_bytes()).hexdigest()!=SHA: raise RuntimeError('trusted reference SHA mismatch')
    m=optimize(qmat_relower(onnx.load(SRC)))
    onnx.checker.check_model(m);m=onnx.shape_inference.infer_shapes(m,strict_mode=True);onnx.save(m,OUT)
    print(f'wrote {OUT}')
if __name__=='__main__': main()
