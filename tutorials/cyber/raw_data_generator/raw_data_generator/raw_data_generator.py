from jinja2 import Environment, PackageLoader, select_autoescape
import os
import json
import logging

class RawDataGenerator:
    def __init__(self):
        # Read templates
        self.env = Environment(loader=PackageLoader('raw_data_generator', 'templates'),)
        self.schema = ["Time", "EventID", "LogHost", "LogonType", "LogonTypeDescription", "UserName", "DomainName", "LogonID",
                       "SubjectUserName", "SubjectDomainName", "SubjectLogonID", "Status", "Source", "ServiceName", "Destination",
                       "AuthenticationPackage", "FailureReason", "ProcessName", "ProcessID", "ParentProcessName", "ParentProcessID", "Raw"]

    # Generates raw data from templates
    def generate_raw_data(self, infilepath, outfilepath, output_format):
        with open(infilepath, "r") as infile:
            filename = os.path.basename(infilepath).split('.')[0]
            logging.info("Reading fileprint... " + infilepath)
            logging.info(outfilepath + '/' + filename + '.' + output_format)
            with open(outfilepath + '/' + filename + '.' + output_format, "w") as outfile:
                logging.info("Writing to file..." +  outfilepath)
                if output_format == "csv":
                    # Write header
                    outfile.write((",".join(self.schema) + "\n"))
                for line in infile:
                    str_line = line
                    json_line = json.loads(str_line)
                    raw_data = repr(self._generate_raw_log(json_line))
                    json_line["Raw"] = raw_data
                    if output_format == "csv":
                        raw_line = self._add_raw_data_to_csv(json_line)
                    else: #json
                        raw_line = json.dumps(json_line)
                    # If this line from the input source ends in a newline, then add a newline to output line
                    if repr(str_line)[-3:] == "\\n'":
                        raw_line = raw_line + "\n"
                    outfile.write(raw_line)
                logging.info("Generate raw data is complete")

    def _generate_raw_log(self, json_line):
        event_code = json_line['EventID']
        event_template = self.env.get_template("event_" + str(event_code) + ".txt")
        return event_template.render(json_line)

    def _add_raw_data_to_csv(self, json_data):
        csv_str = str(json_data["Time"])
        for val in self.schema[1:]:
            data = str(json_data[val]) if val in json_data else ""
            csv_str = csv_str + "," + data
        return csv_str
