#!/bin/bash

# Proxmox GPU Passthrough Setup für NVIDIA P100
# Dieses Script muss auf dem PROXMOX HOST ausgeführt werden (nicht in der VM!)
# Ausführen als root: bash proxmox_gpu_passthrough_setup.sh

echo "=========================================="
echo "Proxmox GPU Passthrough Setup"
echo "=========================================="
echo ""
echo "⚠️  WICHTIG: Dieses Script muss auf dem PROXMOX HOST laufen!"
echo "⚠️  NICHT in der VM ausführen!"
echo ""
read -p "Bist du auf dem Proxmox Host? (ja/nein): " confirm
if [ "$confirm" != "ja" ]; then
    echo "Abgebrochen. Bitte auf dem Proxmox Host ausführen."
    exit 1
fi

# Schritt 1: GPU finden
echo ""
echo "Schritt 1: Suche nach NVIDIA GPU..."
GPU_INFO=$(lspci | grep -i nvidia)
if [ -z "$GPU_INFO" ]; then
    echo "❌ Keine NVIDIA GPU gefunden!"
    echo "Bitte prüfen ob GPU physisch installiert ist."
    exit 1
fi

echo "✓ GPU gefunden:"
echo "$GPU_INFO"
GPU_PCI=$(echo "$GPU_INFO" | awk '{print $1}')
echo "PCI Adresse: $GPU_PCI"

# Schritt 2: IOMMU Status prüfen
echo ""
echo "Schritt 2: Prüfe IOMMU Status..."
if dmesg | grep -q "IOMMU enabled"; then
    echo "✓ IOMMU ist bereits aktiviert"
else
    echo "⚠️  IOMMU ist nicht aktiviert. Konfiguriere jetzt..."
    
    # Backup von GRUB config
    cp /etc/default/grub /etc/default/grub.backup
    
    # Prüfe CPU Typ
    if grep -q "Intel" /proc/cpuinfo; then
        IOMMU_PARAM="intel_iommu=on iommu=pt"
    else
        IOMMU_PARAM="amd_iommu=on iommu=pt"
    fi
    
    # Füge IOMMU Parameter hinzu
    sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"quiet\"/GRUB_CMDLINE_LINUX_DEFAULT=\"quiet $IOMMU_PARAM\"/" /etc/default/grub
    
    update-grub
    echo "✓ IOMMU Parameter hinzugefügt"
    echo "⚠️  REBOOT ERFORDERLICH nach diesem Script!"
fi

# Schritt 3: VFIO Module konfigurieren
echo ""
echo "Schritt 3: Konfiguriere VFIO Module..."

if ! grep -q "vfio" /etc/modules; then
    echo "vfio" >> /etc/modules
    echo "vfio_iommu_type1" >> /etc/modules
    echo "vfio_pci" >> /etc/modules
    echo "vfio_virqfd" >> /etc/modules
    echo "✓ VFIO Module zu /etc/modules hinzugefügt"
else
    echo "✓ VFIO Module bereits konfiguriert"
fi

# Update initramfs
update-initramfs -u -k all
echo "✓ initramfs aktualisiert"

# Schritt 4: VM ID abfragen
echo ""
echo "Schritt 4: GPU an VM durchreichen..."
echo "Verfügbare VMs:"
qm list

read -p "Gib die VM ID ein (z.B. 100): " VM_ID

if ! qm status $VM_ID &>/dev/null; then
    echo "❌ VM $VM_ID nicht gefunden!"
    exit 1
fi

echo "✓ VM $VM_ID gefunden"

# Schritt 5: GPU an VM durchreichen
echo ""
echo "Schritt 5: Reiche GPU durch..."

# Prüfe ob GPU bereits durchgereicht ist
if qm config $VM_ID | grep -q "hostpci"; then
    echo "⚠️  VM hat bereits PCI Passthrough konfiguriert:"
    qm config $VM_ID | grep hostpci
    read -p "Überschreiben? (ja/nein): " overwrite
    if [ "$overwrite" != "ja" ]; then
        echo "Abgebrochen."
        exit 1
    fi
fi

# GPU durchreichen
qm set $VM_ID -hostpci0 $GPU_PCI,pcie=1
echo "✓ GPU an VM $VM_ID durchgereicht"

# Schritt 6: VM neu starten
echo ""
echo "Schritt 6: VM neu starten..."
read -p "VM $VM_ID jetzt neu starten? (ja/nein): " restart_vm

if [ "$restart_vm" = "ja" ]; then
    qm stop $VM_ID
    sleep 3
    qm start $VM_ID
    echo "✓ VM neu gestartet"
fi

# Zusammenfassung
echo ""
echo "=========================================="
echo "Setup abgeschlossen!"
echo "=========================================="
echo ""
echo "Nächste Schritte:"
echo ""
if ! dmesg | grep -q "IOMMU enabled"; then
    echo "1. ⚠️  PROXMOX HOST NEU STARTEN (für IOMMU)"
    echo "   reboot"
    echo ""
fi
echo "2. In der VM einloggen und prüfen:"
echo "   nvidia-smi"
echo ""
echo "3. Falls nvidia-smi nicht funktioniert, in der VM:"
echo "   sudo modprobe nvidia"
echo "   nvidia-smi"
echo ""
echo "4. Falls immer noch nicht funktioniert:"
echo "   lspci | grep -i nvidia"
echo "   (sollte die P100 zeigen)"
echo ""
echo "=========================================="

