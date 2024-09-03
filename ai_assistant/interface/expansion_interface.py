# expansion_interface.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QFileDialog, QListWidgetItem


class ExpansionInterface(QWidget):
    def __init__(self, central_ai):
        super().__init__()
        self.central_ai = central_ai
        self.connected_ais = []
        self.init_ui()


    def init_ui(self):
        layout = QVBoxLayout()

        self.title = QLabel("Connected AIs")
        layout.addWidget(self.title)

        self.ai_list = QListWidget()
        layout.addWidget(self.ai_list)

        self.add_button = QPushButton("Add AI")
        self.add_button.clicked.connect(self.handle_add_ai)
        layout.addWidget(self.add_button)

        self.setLayout(layout)

    def handle_add_ai(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Add AI")
        if file_name:
            try:
                new_ai = self.central_ai.add_ai(file_name)
                self.connected_ais.append(new_ai)
                self.update_ai_list()
            except Exception as e:
                print(f"Error adding AI: {e}")


    def handle_remove_ai(self, ai_id):
        try:
            self.central_ai.remove_ai(ai_id)
            self.connected_ais = [ai for ai in self.connected_ai_id if ai['id'] != ai_id]
            self.update_ai_list()
        except Exception as e:
            print(f"Error removing AI: {e}")

    
    def update_ai_list(self):
        self.ai_list.clear()
        for ai in self.connected_ais:
            item = QListWidgetItem(ai['name'])
            self.ai_list.addItem(item)
            remove_button = QPushButton("Remove")
            remove_button.clicked.connect(lambda: self.handle_remove_ai(ai['id']))
            self.ai_list.setItemWidget(item, remove_button)

            