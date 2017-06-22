"""
    Confusion lattice decoder for the TextCleanser. This module requires SRI-LM toolkit
    in order to function, see the README.
"""

import os
import subprocess
import sys
import re
import select

LATTICE_TOOL_DIR = "/Users/ygorgallina/Documents/Cours/M1/Stage/Outils/srilm-1.6.0/bin/macosx/"
LM_DIR = "/Users/ygorgallina/Documents/Cours/M1/Stage/Outils/TextCleanser/data/"

EMPTY_SYM = "EMPTYSYM"   # placeholder for empty symbol


class Decoder:
    """
        This class decodes a word lattice in pfsg format as output by Generator module's
        generate_word_lattice() function to produce the most likely sentence.
    """

    def __init__(self, port=None, ip=None):
        """ ngram now gets started outside of this module via start_ngram_server.sh script,
        so start_ngram_server() has become obsolete, although it might be a better way..
        """

        if port is None:
            self.port = "12345"
        else:
            self.port = port

        if ip is None:
            self.ip = "127.0.0.1"
        else:
            self.ip = ip

        self.ngram_server_address = "{}@{}".format(self.port, self.ip)
        self.ngram_server_pid = None

        self.is_running = False

        self.decode_command = [LATTICE_TOOL_DIR + "lattice-tool", "-in-lattice", "-", "-read-mesh", "-posterior-decode",
                               "-zeroprob-word", "blemish", "-use-server", self.ngram_server_address]

        # check for presence of "lattice-tool"
        assert (os.path.exists(LATTICE_TOOL_DIR + "lattice-tool"))
        # check that "ngram" exists (for serving the language model)
        assert (os.path.exists(LATTICE_TOOL_DIR + "ngram"))

        # Start ngram server in server mode
        self.start_ngram_server()

    def ngram_server_is_running(self):

        is_running = False

        p = subprocess.Popen(self.decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        r, _, _ = select.select([p.stderr], [], [], 1)
        if p.stderr in r:
            line = p.stderr.readline().decode('utf-8').strip()
            if line == "server {}: Connection refused".format(self.ngram_server_address):
                is_running = False
            else:
                is_running = True
        else:
            is_running = True
        p.kill()
        return is_running


    def start_ngram_server(self):
        if self.ngram_server_is_running():
            self.is_running = True
            sys.stderr.write("SUCCESS : ngram server already launched\n")
            return self.is_running

        sys.stderr.write("Launching ngram server at {}\n".format(self.ngram_server_address))

        textcleanser_root = os.path.split(os.path.realpath(__file__))[0] + os.sep
        command = [textcleanser_root + "start_ngram_server.sh", self.port]

        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        launched = False

        i = 0

        def timeout(x):
            return 120 if x < 2 else 1

        while True:
            r, _, _ = select.select([p.stderr, p.stdout], [], [], timeout(i))
            i += 1
            if p.stderr in r:
                line = p.stderr.readline().decode('UTF-8').strip()
                if line == "starting prob server on port {}".format(self.port):
                    launched = True
                elif line == "could not bind socket: Address already in use":
                    # The server was already launched so the pid is not the goog one
                    self.ngram_server_pid = None
                elif "error" in line:
                    launched = False
            elif p.stdout in r:
                # start_ngram_server.sh outputs the PID of the ngram server process on stdout
                line = p.stdout.readline().decode('UTF-8').strip()
                self.ngram_server_pid = line
            else:
                break

        self.is_running = launched
        if self.is_running:
            sys.stderr.write("SUCCESS : ngram server launched (pid: {})\n".format(self.ngram_server_pid))
        else:
            sys.stderr.write("ERROR : Failed to launch ngram server")
        return self.is_running

    def stop_ngram_server(self):
        if self.ngram_server_pid is None:
            return
        subprocess.Popen(["kill", self.ngram_server_pid])
        print("closing ngram server")
        self.is_running = False

    def close(self):
        self.stop_ngram_server()

    def decode(self, word_mesh):
        """ @param word_mesh string containing word mesh to decode in pfsg format
            @return (stdout,stderr)
        """

        if not self.is_running:
            return None, "Server not running"

        try:
            p = subprocess.Popen(self.decode_command, shell=True, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate(word_mesh.encode('UTF-8'))

            stdout = stdout.decode('UTF-8')
            # if sentence initial marker '<s>' not found, indicates something went wrong
            # during decoding
            if stdout.find("<s>") == -1:
                print("Decoder error output: {}".format(stdout))
                print("word_mesh: {}".format(word_mesh))
                return "", "ERROR"

            # strip out the sentence
            clean_sent = re.sub("- <s>(.*?)</s>", r'\1', stdout)
            # replace empty symbol tokens with ''
            clean_sent = clean_sent.replace(EMPTY_SYM, '')

            # if 'SOMENAME' still present in string, it also indicates something went wrong..
            # NB: 'find' returns starting index of matching substring in str, -1 if not found
            # if clean_sent.find("SOMENAME")!=-1:
            # error
            #   print("Second error.")
            #   print("Decoder error output: {}".format(stdout))
            #   print("word_mesh: ".format(word_mesh))
            #   return "", "ERROR"

            # workaround for bug where p (and its associated pipes) do not
            # seem to be terminated/closed properly
            os.system("killall lattice-tool > /dev/null 2>&1")
            # another possible solution
            # os.close(p.fileno())

            return clean_sent, stderr

        except subprocess.CalledProcessError as ce:
            sys.stderr.write("Error executing command: '{}'\n\n".format(str(ce)))
            return
        except OSError as os_e:     # I think this might be due to ngram server dying
            print("Intercepted OSError => ".format(os_e))
            # restart ngram server
            self.start_ngram_server()
        except ValueError:
            print("Intercepted ValueError.")
            return "ValueError thrown.", "ERROR"

    def __enter__(self):
        self.start_ngram_server()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

if __name__ == "__main__":
    Decoder()
    print("Decoder module.")
