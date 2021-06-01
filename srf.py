import os
from math import ceil


def write(filename, srf):
    """ Write source model to text file in SRF format. 

    Args:
        filename (str): path to file, does not support unix short-cuts (eg., '~')
        srf (Source): data model desribining srf source model

    Returns:
        None
    """
    writer = Writer()
    writer.write(filename, srf)
    return 

def read(filename):
    """ read srf source model in text file to Source object.

    Args:
        filename (str): path to file, does not support unix short-cuts (eg., '~')

    Returns:
        srf (Source): data model desribining srf source model
    """
    reader = Reader()
    srf = reader.read(filename)
    return srf


class FaultSegment:
    """ contains header information for each subfault and contains a list of point source classes that
        describe the information about each subfault on that segment.
    """
    def __init__( self ):
        self.npnts = 0
        self.elon = None
        self.elat = None
        self.nstk = None
        self.ndip = None
        self.len = None
        self.wid = None
        self.stk = None
        self.dip = None
        self.dtop = None
        self.shyp = None
        self.dhyp = None 


class PointSource:
    """ contains header and slip-rate information for each sub-fault. """
    def __init__(self):
        self.lon = None
        self.lat = None
        self.dep = None
        self.stk = None
        self.dip = None
        self.area = None
        self.tinit = None
        self.dt = None
        self.vs = None
        self.den = None
        self.rake = None
        self.slip1 = None
        self.slip2 = None
        self.slip3 = None
        self.nt1 = None
        self.nt2 = None
        self.nt3 = None 
        self.sr1 = []
        self.sr2 = []
        self.sr3 = []

    @property
    def slip(self):
        return (self.slip1, self.slip2, self.slip3)
    

class FiniteFaultSource:
    def __init__(self):
        self.version = None
        self.point_sources = []
        self.segment_headers = []
        # used for iterating
        self._idx = 0

    def __next__(self):
        if self._idx >= len(self.point_sources):
            self._idx = 0
            raise StopIteration
        if self.nsegments > 0:
            val = self.point_sources[self._idx]
            self._idx += 1 
            return val
        else:
            return None

    def __iter__(self):
        return self

    @property
    def has_segment_headers(self):
        if len(self.segment_headers) >= 1:
            return True
        else:
            return False

    @property
    def nsegments(self):
        return len(self.point_sources)

    @property
    def nsources(self):
        return sum(self.point_sources, [])


class Writer:
    def write(self, file, srf):
        """ public function to write srf file """
        self.file = open(file, 'w')
        self.srf = srf
        has_segment_headers = srf.has_segment_headers
        # write global headers
        self._write_global_header()
        # write segment headers, if present
        if has_segment_headers:
            # write plane flag and segments. currently, only supports writing PLANE
            self.file.write(" ".join(['PLANE', str(self.srf.nsegments)])+'\n')
            self._write_segment_header()
        # write data blocks
        for src in srf.point_sources:
            nsrc = len(src)
            self.file.write(" ".join(['POINTS', str(nsrc)])+'\n')
            self._write_data_block(src)
        self.file.write('\n')

    def _write_global_header(self):
        """ writes version number and global header information. """
        if self.srf.version != 2:
            print("Currently, only version 2 is supported.")
        self.file.write(str(self.srf.version)+'\n')

    def _write_segment_header(self):
        """ writes segment header containing sub-fault information """
        for seg in self.srf.segment_headers:
            line_one = [
                f"{seg.elon:.4f}",
                f"{seg.elat:.4f}",
                f"{int(seg.nstk)}",
                f"{int(seg.ndip)}",
                f"{seg.len:.2f}",
                f"{seg.wid:.2f}"
            ]
            self.file.write(" ".join(line_one) + '\n')
            line_two = [
                f"{int(seg.stk)}",
                f"{int(seg.dip)}",
                f"{seg.dtop:.2f}",
                f"{seg.shyp:.2f}",
                f"{seg.dhyp:.2f}"
            ]
            self.file.write(" ".join(line_two) + '\n')
        return
            
    def _write_data_block(self, sources):
        """ Writes data block consisting of source models  """
        # data block contains header rows and then slip rates organized as tables with six column values
        for src in sources:
            line_one = [
                f"{src.lon:.4f}",
                f"{src.lat:.4f}",
                f"{src.dep:.4f}",
                f"{int(src.stk)}",
                f"{int(src.dip)}",
                f"{src.area:1.4e}",
                f"{src.tinit:.4f}",
                f"{src.dt:1.5e}",
                f"{src.vs:1.5e}",
                f"{src.den:1.5e}"
            ]
            self.file.write(" ".join(line_one) + '\n')
            line_two = [
                f"{int(src.rake)}",
                f"{src.slip1:.2f}",
                f"{int(src.nt1)}",
                f"{src.slip2:.2f}",
                f"{int(src.nt2)}",
                f"{src.slip3:.2f}",
                f"{src.nt3}"
            ]
            self.file.write(" ".join(line_two) + '\n')
            # time-series 
            for gidx in range(src.nt1+src.nt2+src.nt3): 
                if gidx < src.nt1:
                    lidx = gidx
                    sr = src.sr1[lidx]
                elif gidx >= src.nt1 and gidx < (src.nt1 + src.nt2):
                    lidx = gidx - src.nt1
                    sr = src.sr2[lidx]
                elif gidx >= (src.nt1 + src.nt2):
                    lidx = gidx - (src.nt1 + src.nt2)
                else:
                    raise ValueError("malformed time-step numbers")
                # write sep can be " " or "\n"
                self.file.write(f"{sr:1.5e}")
                # write newline if 6 or last value
                if gidx != src.nt1 + src.nt2 + src.nt3 - 1:
                    if (gidx + 1) % 6 != 0:
                        self.file.write(" ")
                    else:
                        self.file.write('\n')
                else:
                    self.file.write('\n')


class Reader: 
    """          
    Parses SRF File header and data block.  self.data contains list of FaultSegment classes with header
    information and subfault data contained.
    """          
    def read(self, file):
        """ 
        Main function to read srf format into data structure in python 
        """
        self.file = open(file, 'r')
        self.srf = FiniteFaultSource()
        # read global header information
        self.srf.version = self._read_srf_version()
        line = self.file.readline()
        # read optional comments
        if line.startswith('#'):
            line = self._skip_comments_and_return_next()
        # parse indicator line 
        msg, npts = line.rstrip('\n').split()
        # if fault segment header is present, read headers then data
        npts = int(npts)
        if msg == 'PLANE':
            # segment headers and data blocks are stored in series
            for _ in range(npts):
                seg = self._read_segment_header()
                self.srf.segment_headers.append(seg)
            # headers and data are written in series
            for _ in range(npts):
                msg, nsrcs = self.file.readline().rstrip('\n').split()
                data = self._read_data_block(nsrcs)
                self.srf.point_sources.append(data)
        # if no header is present, read point source data     
        elif msg == 'POINTS':
            data = self._read_data_block(npts)
            self.srf.point_sources.append(data)
        # those are the only two conditions poissible in srf format
        else:
            self.file.close()
            raise IOError('Invalid format for SRF file.')
        self.file.close()
        # if headers are present, ensure that each segment with data has one
        if self.srf.has_segment_headers:
            assert len(self.srf.segment_headers) == len(self.srf.point_sources) 
        return self.srf

    def _skip_comments_and_return_next(self):
        while True:
            try:
                line = self.file.readline()
            except EOFError as e:
                raise EOFError('EOF reached without any data lines. Check formatting of SRF file.')
            if not line.startswith('#'):
                return line

    def _read_srf_version(self):
        """ Reads SRF format version.  Note: Can add sanity checks based on checking for version, etc. """
        # Note: Comma after to not output as list
        version = float(self.file.readline().rstrip('\n').split()[0])
        if version == 2:
            return version
        else:
            raise IOError(f'{self.srf.version} not supported. File must be stored using Version 2.0. File must start with version.')

    def _read_segment_header(self):
        seg = FaultSegment()
        # read header lines
        elon, elat, nstk, ndip, length, wid = self.file.readline().rstrip('\n').split()
        stk, dip, dtop, shyp, dhyp = self.file.readline().rstrip('\n').split()
        # assign header variables
        seg.elon = float(elon)
        seg.elat = float(elat)
        seg.nstk = int(nstk)
        seg.ndip = int(ndip)
        seg.len = float(length)
        seg.wid = float(wid)
        seg.stk = float(stk)
        seg.dip = float(dip)
        seg.dtop = float(dtop)
        seg.shyp = float(shyp)
        seg.dhyp = float(dhyp)
        return seg 

    def _read_data_block(self, npts):
        """ reads the header infomation from the data block """
        npts = int(npts)
        point_sources = []
        for subfault in range(npts):
            sflt = PointSource()
            # read header information
            line1 = self.file.readline().rstrip('\n').split()
            line2 = self.file.readline().rstrip('\n').split()
            lon, lat, dep, stk, dip, area, tinit, dt, vs, den = line1
            # lon, lat, dep, stk, dip, area, tinit, dt, vs, den = self.file.readline().rstrip('\n').split()
            rake, slip1, nt1, slip2, nt2, slip3, nt3 = line2
            # rake, slip1, nt1, slip2, nt2, slip3, nt3 = self.file.readline().rstrip('\n').split()
            # update subfault header
            sflt.lon = float(lon)
            sflt.lat = float(lat)
            sflt.dep = float(dep) # Note: Given in (km)
            sflt.stk = float(stk)
            sflt.dip = float(dip)
            sflt.area = float(area) # Note: Given in (cm^2)
            sflt.tinit = float(tinit)
            sflt.dt = float(dt)
            sflt.vs = float(vs) # Note: Given in (cm/s)
            sflt.den = float(den) # Note: Given in (g/cm^3)
            sflt.rake = float(rake)
            sflt.slip1 = float(slip1)
            sflt.nt1 = int(nt1)
            sflt.slip2 = float(slip2)
            sflt.nt2 = int(nt2)
            sflt.slip3 = float(slip3)
            sflt.nt3 = int(nt3)
            # note: slip rate tables in SRF format are 6 rows across
            nt_all = sflt.nt1 + sflt.nt2 + sflt.nt3
            stf_lines = int(ceil(nt_all / 6.0))
            # read slip rates
            count = 0
            for line_num in range(stf_lines):
                line = self.file.readline().rstrip('\n').split()
                for sr in line:
                    if count < sflt.nt1:
                        sflt.sr1.append(float(sr))
                    elif count >= sflt.nt1 and count < (sflt.nt1+sflt.nt2):
                        sflt.sr2.append(float(sr))
                    elif count >= sflt.nt2 and count < (sflt.nt1+sflt.nt2+sflt.nt3):
                        sflt.sr3.append(float(sr))
                    count+=1
            # append point source
            point_sources.append(sflt)
        return point_sources


if __name__ == '__main__':

    def comp_files(f1, f2):
        """ compares two files line by line ignoring whitespace """
        f1h = open(f1, 'r')
        f2h = open(f2, 'r')

        f1_val = []
        f2_val = []
        for l1, l2 in zip(f1h, f2h):
            if not l1.startswith('#'):
                l1_clean = l1.rstrip('\n').split()
                f1_val.append(l1_clean)
            if not l2.startswith('#'):
                l2_clean = l2.rstrip('\n').split()
                f2_val.append(l2_clean)
        for lnum, (l1, l2) in enumerate(zip(f1_val, f2_val)):
            if l1 == l2:
                continue
            else:
                print('>>> ' + " ".join(l1))
                print('<<< ' + " ".join(l2))

    files = ['./examples/point_source.srf',
             './examples/single_planar_fault.srf',
             './examples/multi_segment_rupture.srf']

    for i, f in enumerate(files):
        # test point source
        print(f'Testing {f}...')
        src = read(f)
        write(f'/tmp/test{i}.srf', src)
        comp_files(f'/tmp/test{i}.srf', f)
    
