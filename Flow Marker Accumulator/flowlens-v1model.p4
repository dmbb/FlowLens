/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

/*Set number of shifts according to the quantization level
    QL=2,    1
    QL=4,    2
    QL=8,    3
    QL=16,   4
    QL=32,   5
    QL=64,   6
    QL=128,  7
    QL=256,  8
*/

/* In our running example, we will use QL=16 */
const bit<8> BIN_WIDTH_SHIFT = 4; 

/* Number of counters for each flow */
const bit<32> FLOW_BINS = 1500 >> BIN_WIDTH_SHIFT; //94 flow counters for QL=16

/* Number of flows in each partition */
const bit<32> FLOWS_PER_PARTITION = 3000;

const bit<32> PARTITION_SIZE = FLOWS_PER_PARTITION*FLOW_BINS;

/* Number of packet sizes considered for truncation */
const bit<32> NUM_PKT_SIZES = 1500;

/* To flag bins not to be counted */
const bit<1> NOBIN_FLAG = 0; 


typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;
const bit<16> TYPE_IPV4 = 0x800;
typedef bit<8> ip_protocol_t;
const ip_protocol_t IP_PROTOCOLS_TCP = 6;
const ip_protocol_t IP_PROTOCOLS_UDP = 17;

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

//User-defined metadata 
struct metadata {
    bit  truncation_flag; // marks whether or not the current pkt has to be counted
    bit<32> rg_bin_offset; // this is computed by adding the binIndex_posTruncation to the flow_offset
	bit<32> binIndex_preTruncation;
	bit<32> binIndex_posTruncation;
}

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;
    udp_t        udp;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    // Initial state of the parser
    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6: parse_tcp;
            17: parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }
}


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {   
    apply {  }
}

/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }
    

    ///////////////////////////////////////////////////////
    //Set ipv4 forwarding for packets traversing the switch
    ///////////////////////////////////////////////////////
    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        standard_metadata.egress_spec = port;     //Sets the egress port for the next hop. 
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;  //Updates the ethernet destination address with the address of the next hop. 
        hdr.ethernet.dstAddr = dstAddr; //Updates the ethernet source address with the address of the switch.
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;  //Decrements time to live
    }


    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
        }
        actions = {
            ipv4_forward;
            drop;
        }
        size = 1024;
        default_action = drop();
    }
    
    apply {

        if (hdr.ipv4.isValid()) {
            
            ipv4_lpm.apply();
            
        }
		else {
			drop();
		}
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {

    register<bit<16>>(PARTITION_SIZE) reg_grid0;
    register<bit<16>>(PARTITION_SIZE) reg_grid1;
    register<bit<16>>(PARTITION_SIZE) reg_grid2;
    register<bit<16>>(PARTITION_SIZE) reg_grid3;
    register<bit<16>>(PARTITION_SIZE) reg_grid4;
    register<bit<16>>(PARTITION_SIZE) reg_grid5;
    register<bit<16>>(PARTITION_SIZE) reg_grid6;
    register<bit<16>>(PARTITION_SIZE) reg_grid7;
    register<bit<16>>(PARTITION_SIZE) reg_grid8;


	//****************** Register Actions Definition************************
    action reg_grid0_action() { 
        bit<16> value;  
        reg_grid0.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid0.write(meta.rg_bin_offset, value);
    }

    action reg_grid1_action() { 
        bit<16> value;  
        reg_grid1.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid1.write(meta.rg_bin_offset, value);
    }

    action reg_grid2_action() { 
        bit<16> value;  
        reg_grid2.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid2.write(meta.rg_bin_offset, value);
    }

    action reg_grid3_action() { 
        bit<16> value;  
        reg_grid3.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid3.write(meta.rg_bin_offset, value);
    }

    action reg_grid4_action() { 
        bit<16> value;  
        reg_grid4.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid4.write(meta.rg_bin_offset, value);
    }

    action reg_grid5_action() { 
        bit<16> value;  
        reg_grid5.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid5.write(meta.rg_bin_offset, value);
    }

    action reg_grid6_action() { 
        bit<16> value;  
        reg_grid6.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid6.write(meta.rg_bin_offset, value);
    }

    action reg_grid7_action() { 
        bit<16> value;  
        reg_grid7.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid7.write(meta.rg_bin_offset, value);
    }

    action reg_grid8_action() { 
        bit<16> value;  
        reg_grid8.read(value, meta.rg_bin_offset);
        value = value+1;
        reg_grid8.write(meta.rg_bin_offset, value);
    }

	//******************End Register Actions Definition*********************

	//****************** Other Actions Definition************************

    // flow_offset: is used for indexing the flow within a bin of the reg_grid 
    action set_flow_data(bit<32> flow_offset) {
		meta.rg_bin_offset = flow_offset + meta.binIndex_posTruncation;
    }

	action quantization_act(){
		meta.binIndex_preTruncation =  (bit<32>) (standard_metadata.packet_length >> BIN_WIDTH_SHIFT);
	}

    action truncate_binIndex(bit<32> new_index, bit flag) {
        meta.binIndex_posTruncation = new_index;
        meta.truncation_flag = flag;
    }


	//******************End Other Actions Definition*********************

	//******************Tables Definition**************************

    table flow_tbl0 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl1 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl2 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl3 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl4 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl5 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl6 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl7 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table flow_tbl8 {
        key = {
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.srcAddr: exact;
			meta.truncation_flag : exact;
        }
        actions = {
            set_flow_data;
            NoAction();
        }
        default_action = NoAction();
		size = FLOWS_PER_PARTITION;
    }

    table truncation_tbl {
        key = {
            meta.binIndex_preTruncation: exact;
        }
        actions = {
            truncate_binIndex();
            NoAction();
        }
        default_action = truncate_binIndex(0, NOBIN_FLAG);
		size = NUM_PKT_SIZES;
    }


	//******************End Tables Definition***********************


    apply {

			quantization_act();

			truncation_tbl.apply();

            if(flow_tbl0.apply().hit) {
             	reg_grid0_action();
			}	
			else {
			 if(flow_tbl1.apply().hit) {
             	reg_grid1_action();
			 }
			 else {
			  if(flow_tbl2.apply().hit) {
             	reg_grid2_action();
			  }
			  else {
			  	if(flow_tbl3.apply().hit) {
              		reg_grid3_action();
			  	}
				else {
			  	 if(flow_tbl4.apply().hit) {
              	 	reg_grid4_action();
			  	 }
				 else {
			  	  if(flow_tbl5.apply().hit) {
              	  		reg_grid5_action();
			  	  }
				  else {
			  	   if(flow_tbl6.apply().hit) {
              	   		reg_grid6_action();
			  	   }
				   else {
			  	    if(flow_tbl7.apply().hit) {
              	    	reg_grid7_action();
			  	    }
					else {
			  	     if(flow_tbl8.apply().hit) {
              	     	reg_grid8_action();
			  	     }
					}
				   }
				  }
				 }
				}
			  }
			 }
			}
			       
    } // end of the apply block

}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	      	  hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}


/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {

    //deparser that selects the order in which fields inserted into the outgoing packet.
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
