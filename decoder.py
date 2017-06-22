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

TEXTCLEANSER_ROOT = os.path.split(os.path.realpath(__file__))[0] + os.sep


class Decoder:
    """
        This class decodes a word lattice in pfsg format as output by Generator module's
        generate_word_lattice() function to produce the most likely sentence.
    """

    def __init__(self, port=None, ip=None):
        """ ngram now gets started outside of this module via start_ngram_server.sh script,
        so start_ngram_server() has become obsolete, although it might be a better way..
        """

        self.port = "12345" if port is None else port
        self.ip = "127.0.0.1" if ip is None else ip

        self.ngram_server_address = "{}@{}".format(self.port, self.ip)
        self.ngram_server_process = None

        self.is_running = False

        self.decode_command = [LATTICE_TOOL_DIR + "lattice-tool", "-in-lattice", "-", "-read-mesh", "-posterior-decode",
                               "-zeroprob-word", "blemish", "-use-server", self.ngram_server_address]
        self.start_server_command = [LATTICE_TOOL_DIR + "ngram", "-lm", LM_DIR + "tweet-lm.gz", "-mix-lm",
                                     LM_DIR + "latimes-lm.gz", "-lambda", "0.7", "-mix-lambda2", "0.3",
                                     "-server-port", self.port]

        # check for presence of "lattice-tool"
        assert (os.path.exists(LATTICE_TOOL_DIR + "lattice-tool"))
        # check that "ngram" exists (for serving the language model)
        assert (os.path.exists(LATTICE_TOOL_DIR + "ngram"))

        # Start ngram server in server mode
        self.start_ngram_server()

    def ngram_server_is_running(self):

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

        p = subprocess.Popen(self.start_server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.ngram_server_process = p

        launched = False
        i = 0

        def timeout(x):
            return 120 if x < 1 else 1

        while True:
            r, _, _ = select.select([p.stderr], [], [], timeout(i))
            i += 1
            if p.stderr in r:
                line = p.stderr.readline().decode('UTF-8').strip()
                if line == "starting prob server on port {}".format(self.port):
                    launched = True
                elif line == "could not bind socket: Address already in use":
                    # The server was already launched so the pid is not the good one
                    self.ngram_server_process = None
                    launched = True
                elif "error" in line:
                    launched = False
            else:
                break

        self.is_running = launched
        if self.is_running:
            sys.stderr.write("SUCCESS : ngram server launched (pid: {})\n".format(self.ngram_server_process))
        else:
            sys.stderr.write("ERROR : Failed to launch ngram server")
        return self.is_running

    def stop_ngram_server(self):
        if self.ngram_server_process is None:
            return
        self.ngram_server_process.kill()
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
            p = subprocess.Popen(" ".join(self.decode_command), shell=True, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate(input=word_mesh.encode('UTF-8'))

            p.kill()

            stdout = stdout.decode('UTF-8')
            stderr = stderr.decode('UTF-8')
            print("Sortie decoddeur")
            print(stdout)
            print(stderr)
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
            # if clean_sent.find("SOMENAME")!=-1:
            # error
            #   print("Second error.")
            #   print("Decoder error output: {}".format(stdout))
            #   print("word_mesh: ".format(word_mesh))
            #   return "", "ERROR"

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
